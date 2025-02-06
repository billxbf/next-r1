import re
from trl import DataCollatorForCompletionOnlyLM

R1PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user \
with the answer. The reasoning process and answer are enclosed within <think> </think> and \
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> \
<answer> answer here </answer>. User: Solve the following math problem, whose answer is an integer. Please reason step by step, and put your final answer within \\boxed{{}}.
{problem} 
Assistant: <think>\n"""


def format_prompt(example):
    output_text = []
    for i in range(len(example["problem"])):
        text = R1PROMPT.format(problem=example["problem"][i])
        output_text.append(text)
    return output_text


def get_trainer_collator(tokenizer):
    response_template = "Assistant: <think>"
    return DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


def extract_answer(answer):
    boxed_pattern = r'\\boxed{(\d+)}'
    boxed_match = re.search(boxed_pattern, answer)

    if boxed_match:
        return int(boxed_match.group(1))

    # If no boxed number found, find the last number in the text
    number_pattern = r'\d+'
    numbers = re.findall(number_pattern, answer)

    if numbers:
        return int(numbers[-1])

    return 0
