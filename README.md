# next-r1
One step beyond Deepseek R1 on Mathematical Reasoning.


## Plan
1. OTS eval of `qwen32r` q4 with vllm, with consensus.
2. Specialization: Collect hard math dataset and use R1 for rejection sampling + SFT
    - Collect QA data from huggingface, AoPS, etc.
    - Use R1 for data labeling ([distilabel](https://github.com/argilla-io/distilabel))
    - Rejection Sampling + SFT, producing `qwen32r-distill`.
3. Exploration (RL): GRPO ([Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer))
    - Custom reward function (accuracy + length penalty?)
    - Expand `qwen32b-distill-math` potentials in reasoning space and produce `qwen32r-distill-rl`.

4. Exploitation: Further supervise on selected trajectories.
    - Collect synthetic data from `qwen32r-distill-rl`.
    - SFT/DPO on filtered generations that's
        - accurate
        - succinct 
        - truthful 

        and produce `qwen32r-distill-rl-sft`



# Experiments

#### dsqwen7b-awq
- cons16, 8k: 0.5 (11min)
- cons16, 16k: 0.5 (44min)

#### dsqwen14b-sft-awq
- cons32, 8k:  0.4 (25min)
- cons32, 16k: 0.7 (82min)
- cons8, 8k: 0.4 (14min)
- cons8, 16k: 0.6 (53min)