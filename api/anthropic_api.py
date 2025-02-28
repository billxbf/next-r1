"""
Script for generating data with Anthropic API, optimized with thread-parallelism and token estimations.
Make sure to set ANTHROPIC_API_KEY in your environment variables.
"""
import os
import anthropic
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
                    model_name="claude-3-7-sonnet-20250219",
                    max_tokens=4096,
                    thinking_budget = 2048,
                    return_raw_object=False,
                    **anthropic_args):

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    try:
        completion = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            # thinking={
            #     "type": "enabled",
            #     "budget_tokens": thinking_budget,
            # },
            messages=[{
                "role": "user",
                "content": instruction
            }],
            **anthropic_args
        )
        if return_raw_object:
            return completion
        response = completion.content[0].text
        prefix, code = response.rsplit("```python\n", 1)
        # Check if the prefix starts with a '#' and remove the first line if it does
        if prefix.lstrip().startswith('#'):
            prefix = '\n'.join(prefix.split('\n')[1:])
        code = "```python\n" + code
        full_response = "<think>\nAlright, let's think through how to approach this problem algorithmically.\n\n" + \
              prefix + "\n</think>\n\n" + code
        return _format_response([instruction], [full_response])
    except Exception as e:
        if return_raw_object:
            return e  # throw the error
        return _format_response([instruction], [f"Error: {e}"])


def generate_batch(instructions,
                   batch_size=8,
                   model_name="claude-3-7-sonnet-20250219",
                   estimate=True,
                   estimate_n_sample=1,
                   max_tokens=4096,
                   save_intermediate=True,
                   **anthropic_args):

    # estimate the number of tokens with small samples
    if estimate:
        test_tokens_completions = []
        test_tokens_prompts = []
        print(f">> Estimating token usage with {estimate_n_sample} samples...")
        for _ in range(estimate_n_sample):
            test_completion = generate_single(random.choice(
                instructions), model_name, max_tokens, return_raw_object=True, **anthropic_args)
            if isinstance(test_completion, Exception):
                raise test_completion
            else:
                test_tokens_completions.append(
                    test_completion.usage.output_tokens)
                test_tokens_prompts.append(test_completion.usage.input_tokens)

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
            args_iter = zip(batch, repeat(model_name),
                            repeat(max_tokens), repeat(False))
            kwargs_iter = repeat(anthropic_args)
            batch_results = _starmap_with_kwargs(
                pool, generate_single, args_iter, kwargs_iter)
            results.extend(
                [item for sublist in batch_results for item in sublist])

            if save_intermediate:
                pkl.dump(results, open("generations.pkl", "wb"))

    return results
    