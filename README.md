# next-r1
One step beyond Deepseek R1 on Mathematical Reasoning.


## Plan
1. OTS eval of `qwen32b-distill` q4 with vllm, with consensus.
2. Specialization: Collect hard math dataset and use R1 for rejection sampling + SFT
    - Collect QA data from huggingface, AoPS, etc.
    - Use R1 for data labeling ([distilabel](https://github.com/argilla-io/distilabel))
    - Rejection Sampling + SFT, producing `qwen32b-distill-math`.
3. Exploration (RL): GRPO ([Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer))
    - Custom reward function (accuracy + length penalty?)
    - Expand `qwen32b-distill-math` potentials in reasoning space and produce `qwen32b-distill-math-rl`.

4. Exploitation: Further supervise on selected trajectories.
    - Collect synthetic data from `qwen32b-distill-math-rl`.
    - SFT/DPO on filtered generations that's
        - accurate
        - succinct 
        - truthful 

        and produce `qwen32b-distill-math-rl-sft`