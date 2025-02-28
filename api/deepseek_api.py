"""
Script for generating data with OpenAI API, optimized with thread-parallelism and token estimations.
Make sure to set DEEPSEEK_API_KEY in your environment variables.
"""
import os
from openai import OpenAI
from tqdm import tqdm
import random
import multiprocessing
from itertools import repeat
import pickle as pkl


def _format_response(instructions, responses):
    assert len(instructions) == len(responses)
    return [{"instruction": instructions[i], "completion": responses[i]} for i in range(len(instructions))]


def _apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


def _starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(_apply_args_and_kwargs, args_for_starmap)


def generate_single(instruction,
                    model_name="deepseek-reasoner",
                    return_raw_object=False,
                    url="https://api.deepseek.com",
                    **openai_args):
    
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=url)
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": instruction}],
            max_tokens=8000,
            **openai_args
        )
        # print(completion)
        if return_raw_object:
            return completion
        response = completion.choices[0].message.content
        reasoning_content = completion.choices[0].message.reasoning_content
        full_response = "<think>\n" + reasoning_content + "\n</think>\n\n" + response
        return _format_response([instruction], [full_response])
    except Exception as e:
        if return_raw_object:
            return e  # throw the error
        return _format_response([instruction], [f"Error: {e}"])


def generate_batch(instructions,
                   batch_size=8,
                   model_name="deepseek-reasoner",
                   estimate=True,
                   estimate_n_sample=1,
                   save_intermediate=True,
                   **openai_args):

    # estimate the number of tokens with small samples
    if estimate:
        test_tokens_completions = []
        test_tokens_prompts = []
        print(f">> Estimating token usage with {estimate_n_sample} samples...")
        for _ in range(estimate_n_sample):
            test_completion = generate_single(random.choice(
                instructions), model_name, return_raw_object=True, **openai_args)
            if isinstance(test_completion, Exception):
                raise test_completion
            else:
                test_tokens_completions.append(
                    test_completion.usage.completion_tokens)
                test_tokens_prompts.append(test_completion.usage.prompt_tokens)

        estimate_tokens_completion = sum(
            test_tokens_completions) // len(test_tokens_completions) * len(instructions)
        estimate_tokens_prompt = sum(
            test_tokens_prompts) // len(test_tokens_prompts) * len(instructions)
        print(
            f"Estimated total token usage: {estimate_tokens_prompt} prompt tokens; {estimate_tokens_completion} completion tokens")

    print(
        f">> Generating {len(instructions)} instructions with batch size {batch_size}...")

    results = []
    with multiprocessing.Pool() as pool:
        for i in tqdm(range(0, len(instructions), batch_size)):
            batch = instructions[i:i+batch_size]
            args_iter = zip(batch, repeat(model_name), repeat(False))
            kwargs_iter = repeat(openai_args)
            batch_results = _starmap_with_kwargs(
                pool, generate_single, args_iter, kwargs_iter)
            results.extend(
                [item for sublist in batch_results for item in sublist])

            if save_intermediate:
                pkl.dump(results, open("generations.pkl", "wb"))

    return results
    