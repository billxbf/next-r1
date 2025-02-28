import re
from collections import Counter
import sys
import subprocess
import random
import os

SPLIT = "<｜Assistant｜>"

# FUSE_PROMPT = """Solve the following math problem. Think creatively, verify carefully and answer directly. The answer should be an integer. 
# ```
# {problem}
# ```
# Use either of the following methods:
# 1. Reason step by step, and put your final answer within \\boxed{{}}. 
# 2. You can also leverage a Python program (eg. with sympy) to solve it. Enclose your solution with fenced code block (```python and ```), and print the answer at last.
# """

EN_PROMPT = """Solve the following AIME math problem. The answer should be an integer. 
```
{problem}
```
Reason step by step, and put your final answer within \\boxed{{}}. 
"""

CN_PROMPT = """求解下述奥林匹克竞赛题，最终答案是一个整数。
```
{problem}
```
请逐步推理，并将最终答案写在 \\boxed{{}} 中。
"""

CODE_PROMPT = """Leverage a Python program (eg. with sympy) to solve the following math problem algorithmically. The answer should be an integer. 
```
{problem}
```
Enclose your solution with fenced code block (```python and ```), and print the answer at last.
"""

def format_prompt(tokenizer, problem, prompt_template):
    chat = [{"role": "user", "content": prompt_template.format(problem=problem)},]
    prompt = tokenizer.apply_chat_template(
        conversation=chat,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt

def naive_parse(answer):
    number_pattern = r'\d+'
    numbers = re.findall(number_pattern, answer)
    if numbers:
        return int(numbers[-1])
    return -1

def extract_answer(output):
    
    if '```python' in output:
        os.makedirs("tmpcode/", exist_ok=True)
        file_id = random.randint(0, 1000000)
        try:
            code = output.split('```python')[-1].split("```")[0]
            code = code.replace('\n', '\n    ')
            # Add a try...except block
            code = "\ntry:\n    from sympy import *\n{}\nexcept Exception as e:\n    print(e)\n    print('FAIL')\n".format(code)
            with open(f'tmpcode/code_{file_id}.py', 'w') as fout:
                fout.write(code)

            batcmd = 'timeout 7 ' + sys.executable + f' tmpcode/code_{file_id}.py'
            try:
                shell_output = subprocess.check_output(batcmd, shell=True).decode('utf8')
                shell_output = shell_output.strip().split('\n')[-1]
                code_output = round(float(eval(shell_output))) % 1000
            except:
                code_output = -1
        
        except Exception as e:
            code_output = -1
    else:
        code_output = -1
    
    try:
        result_output = re.findall(r'\\boxed\{(.*)\}', output)
        if not len(result_output):
            result_output = naive_parse(output)
        else:
            result_output = result_output[-1]

        if not len(result_output):
            result_output = -1
        else:
            result_output = round(float(eval(result_output))) % 1000
    
    except:
        result_output = -1
    # print('CODE RESULTS', code_output, '| BOXED RESULTS', result_output)
    if code_output != -1:
        return code_output
    if result_output != -1:
        return result_output
    else:
        return 250


def select_answer(answers):
    counter = Counter()
    # longer (later) answers have more say.
    beta = 0.0
    for answer in answers:
        try:
            counter[int(answer)] += 1 + beta
            beta += 0.01
        except:
            pass
    if not counter:
        return 920
    _, answer = sorted([(v, k) for k, v in counter.items()], reverse=True)[0]
    return answer % 1000
