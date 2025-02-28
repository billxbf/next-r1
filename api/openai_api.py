"""
Script for generating data with OpenAI API, optimized with thread-parallelism and token estimations.
Make sure to set OPENAI_API_KEY in your environment variables.
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
                    model_name="claude-3-7-sonnet-20250219",
                    sys_prompt="You are a helpful assistant.",
                    return_raw_object=False,
                    **openai_args):
    """
    Generate a single completion for a given instruction using OpenAI's API.

    Args:
        instruction (str): The instruction to generate a completion for.
        model_name (str, optional): The name of the OpenAI model to use. Defaults to "gpt-4o-mini".
        sys_prompt (str, optional): The system prompt to use for the completion. Defaults to "You are a helpful assistant".
        return_raw_object (bool, optional): Whether to return the raw API response object. Defaults to False.
        **openai_args: Additional keyword arguments to pass to the OpenAI API.

    Returns:
        list or object: If return_raw_object is False, returns a list containing a single dictionary with
        "instruction" and "completion" keys. If return_raw_object is True, returns the raw API response object.
        In case of an error, returns a list with a single dictionary containing the error message as the completion.

    Example:
        result = generate_single("What is the capital of France?", model_name="gpt-3.5-turbo", max_tokens=50)
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": instruction}],
            **openai_args
        )
        response = completion.choices[0].message.content
        if return_raw_object:
            return completion
        return _format_response([instruction], [response])
    except Exception as e:
        if return_raw_object:
            return e  # throw the error
        return _format_response([instruction], [f"Error: {e}"])


def generate_batch(instructions,
                   batch_size=8,
                   model_name="gpt-4o-mini",
                   sys_prompt="You are a helpful assistant.",
                   estimate=True,
                   estimate_n_sample=1,
                   save_intermediate=True,
                   **openai_args):
    """
    Generate a batch of completions for given instructions using OpenAI's API.

    Args:
        instructions (list): A list of instruction strings to generate completions for.
        batch_size (int, optional): The number of instructions to process in each batch. Defaults to 8.
        model_name (str, optional): The name of the OpenAI model to use. Defaults to "gpt-4o-mini".
        sys_prompt (str, optional): The system prompt to use for each completion. Defaults to "You are a helpful assistant".
        estimate (bool, optional): Whether to estimate token usage before generation. Defaults to True.
        estimate_n_sample (int, optional): The number of samples to use for token usage estimation. Defaults to 1.
        save_intermediate (bool, optional): Whether to save intermediate results to a "generations.pkl". Defaults to True.
        **openai_args: Additional keyword arguments to pass to the OpenAI API.

    Returns:
        list: A list of dictionaries, each containing an "instruction" and its corresponding "completion".

    Example:
        results = generate_batch(["What is the meaning of life?"]*100, model_name="gpt-4o-mini", estimate_n_sample=10, max_tokens=128, temperature=0.0)

    Note:
        This function uses multiprocessing to generate completions in parallel, which can significantly
        speed up the process for large batches of instructions. Mind the cost.
    """

    # estimate the number of tokens with small samples
    if estimate:
        test_tokens_completions = []
        test_tokens_prompts = []
        print(f">> Estimating token usage with {estimate_n_sample} samples...")
        for _ in range(estimate_n_sample):
            test_completion = generate_single(random.choice(
                instructions), model_name, sys_prompt, return_raw_object=True, **openai_args)
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
            args_iter = zip(batch, repeat(model_name),
                            repeat(sys_prompt), repeat(False))
            kwargs_iter = repeat(openai_args)
            batch_results = _starmap_with_kwargs(
                pool, generate_single, args_iter, kwargs_iter)
            results.extend(
                [item for sublist in batch_results for item in sublist])

            if save_intermediate:
                pkl.dump(results, open("generations.pkl", "wb"))

    return results
    