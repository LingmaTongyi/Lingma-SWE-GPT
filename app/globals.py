"""
Values of global configuration variables.
"""

from typing import Optional

# Overall output directory for results
output_dir: str = ""

# whether to start conversation from fresh, or load from a conversation history.
# If None, start from fresh.
# If not None, continue from the conversation history stored in <file>.
# <file> is the value of this variable, and should points to a json file
# containing the past conversation history.
load_cache: Optional[str] = None

# the model to use
model: str = "gpt-3.5-turbo-0125"

# the model temperature to use
# For OpenAI models: this value should be from 0 to 2
model_temperature: float = 0.0

# upper bound of the number of conversation rounds for the agent
conv_round_limit: int = 15

# whether to perform sbfl
enable_sbfl: bool = False
# whether mtcs
# enable_mtcs: bool = True
enable_mtcs: bool = False
only_mcts: bool = False

# agentless patch samples
multi_patch_sample: bool = False
sample_size: int = 5

# review_repo
review_repo: bool = True

# start from cache.
continue_task_from_cache: bool = False
# continue_task_from_cache: bool = True

# whether to perform layered search
enable_layered: bool = False

# whether to perform our own validation
enable_validation: bool = False
is_pre_test: bool = False

# whether to do angelic debugging
enable_angelic: bool = False

# whether to do perfect angelic debugging
enable_perfect_angelic: bool = False

# perform installation of dependencies for each task to save time
# This should be figured out automatically depending on the value of other options
do_install: bool = False

# A special mode to only save SBFL result and exit
only_save_sbfl_result: bool = False

# timeout for test cmd execution, currently set to 5 min
test_exec_timeout: int = 300

### Some information about the allowed models

MODEL_NOTES = {
    "gpt-4-0125-preview": "Turbo. Up to Dec 2023.",
    "gpt-4-1106-preview": "Turbo. Up to Apr 2023.",
    "gpt-3.5-turbo-0125": "Turbo. Up to Sep 2021.",
    "gpt-3.5-turbo-1106": "Turbo. Up to Sep 2021.",
    "gpt-3.5-turbo-16k-0613": "Turbo. Deprecated. Up to Sep 2021.",
    "gpt-3.5-turbo-0613": "Turbo. Deprecated. Only 4k window. Up to Sep 2021.",
    "gpt-4-0613": "Not turbo. Up to Sep 2021.",
    "gpt-4o-2024-05-13": "Up to Oct 2023",
    "gemini-1.5-pro-latest": "Turbo. Up to Sep",
    "claude-3-opus-20240229": "Turbo. Up to Sep",
    "claude-3-5-sonnet-20240620": "Turbo. Up to Sep",
    "qwen2-72b-instruct": "qwen2-72b-instruct",
    "qwen2-7b-instruct": "qwen2-7b-instruct",
    "pre-Meta-Llama-3.1-405B-Instruct-FP8": "pre-Meta-Llama-3.1-405B-Instruct-FP8",
    "pre-Meta-Llama-3.1-70B-Instruct": "pre-Meta-Llama-3.1-70B-Instruct",
    "pre-Mistral-Large-Instruct-2407": "pre-Mistral-Large-Instruct-2407",
    "qwen2.5-72b-instruct": "qwen2.5-72b-instruct",
    "Lingma-SWE-GPT": "Lingma-SWE-GPT",
}

MODELS = list(MODEL_NOTES.keys())

# The cost per token for each model input.
MODEL_COST_PER_INPUT = {
    "gpt-4-0125-preview": 0.00001,
    "gpt-4-1106-preview": 0.00001,
    "gpt-3.5-turbo-0125": 0.0000005,  # cheapest!
    "gpt-3.5-turbo-1106": 0.000001,
    "gpt-3.5-turbo-16k-0613": 0.000003,
    "gpt-3.5-turbo-0613": 0.0000015,
    "gpt-4-0613": 0.00003,  # most expensive
    "gpt-4o-2024-05-13": 0.000005,
    "gemini-1.5-pro-latest": 0.00003,
    "claude-3-opus-20240229": 0.00003,
    "claude-3-5-sonnet-20240620": 0.000003,
    "qwen2-72b-instruct":0.000005,
    "qwen2-7b-instruct": 0.0,
    "pre-Llama-3.1-70B-Instruct": 0.000005,
    "pre-Meta-Llama-3.1-405B-Instruct-FP8": 0.0,
    "pre-Meta-Llama-3.1-70B-Instruct": 0.0,
    "pre-Mistral-Large-Instruct-2407": 0.0,
    "qwen2.5-72b-instruct": 0.0,
    "Lingma-SWE-GPT": 0.0,
}

# The cost per token for each model output.
MODEL_COST_PER_OUTPUT = {
    "gpt-4-0125-preview": 0.00003,
    "gpt-4-1106-preview": 0.00003,
    "gpt-3.5-turbo-0125": 0.0000015,
    "gpt-3.5-turbo-1106": 0.000002,
    "gpt-3.5-turbo-16k-0613": 0.000004,
    "gpt-3.5-turbo-0613": 0.000002,
    "gpt-4-0613": 0.00006,
    "gpt-4o-2024-05-13": 0.000015,
    "gemini-1.5-pro-latest": 0.00003,
    "claude-3-opus-20240229": 0.00003,
    "claude-3-5-sonnet-20240620": 0.000015,
    "qwen2-72b-instruct": 0.0000015,
    "qwen2-7b-instruct": 0.0,
    "pre-Llama-3.1-70B-Instruct": 0.000015,
    "pre-Meta-Llama-3.1-405B-Instruct-FP8": 0.0,
    "pre-Meta-Llama-3.1-70B-Instruct": 0.0,
    "pre-Mistral-Large-Instruct-2407": 0.0,
    "qwen2.5-72b-instruct": 0.0,
    "Lingma-SWE-GPT": 0.0,
}

# models that support the new parallel tool call feature
PARALLEL_TOOL_CALL_MODELS = [
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-4o-2024-05-13",
    "gemini-1.5-pro-latest",
    "claude-3-opus-20240229",
    "claude-3-5-sonnet-20240620",
    "qwen2-72b-instruct",
    "qwen2-7b-instruct",
    "pre-Llama-3.1-70B-Instruct",
    "pre-Meta-Llama-3.1-405B-Instruct-FP8",
    "pre-Meta-Llama-3.1-70B-Instruct",
    "pre-Mistral-Large-Instruct-2407",
    "qwen2.5-72b-instruct",
    "Lingma-SWE-GPT",
]
