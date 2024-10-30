# Lingma SWE-GPT(SWESynInfer): SoftWare Engineering Process Data Synthesis and Inference Workflow for Lingma SWE-GPT

## Overview


**Lingma SWE-GPT**:  is an open-source large language model specifically designed for software improvement. Built upon the foundation of the Qwen series base models, Lingma SWE-GPT has undergone additional training using software engineering development process data to enhance its capabilities in solving complex software engineering tasks.



**SWESynInfer**: three-stage software engineering process data synthesis and inference workflow. This workflow extends the publicly available AutoCodeRover framework. AutoCodeRover provides baseline processes for context retrieval and patch generation stages, our work further introduces crucial enhancements to more accurately simulate the cognitive processes of expert developers.


## Model Introduction

Lingma SWE-GPT is a specialized model that focuses on addressing the unique challenges faced in software engineering. By leveraging the robust capabilities of the Qwen base models and incorporating domain-specific knowledge, this model aims to provide intelligent assistance across various aspects of software development.

![image](https://github.com/user-attachments/assets/c2b0b1d6-0cc9-42f1-abc3-ed51e52b457c)


## Model Performance

Lingma SWE-GPT has demonstrated impressive performance in software engineering tasks:

- 🌟 Achieved a **30.2% solution rate on the authoritative SWE-bench Verified** leaderboard for software engineering intelligent agents.
- 🌟 Achieved a **51.16%** fault location success rate on SWE-bench Verified.
- 👑 Outperforms other open-source models of similar scale in software engineering-specific tasks (a
22.76% increase compared to Llama 3.1 405B).

## Quick Start
### Setup
First, create a virtual environment and install the required dependencies.
```
git clone https://github.com/LingmaTongyi/SWESynInfer.git
cd SWESynInfer
conda env create -f environment.yml
conda activate swesyninfer

# Set repo_path in setup_map.json (SWESynInfer/SWE-bench/setup_result/setup_map.json) to the local path
python scripts/1_change_testbed_path.py YOUR_ABSOLUTE_PATH/SWESynInfer/SWE-bench/repos/testbed
```
### Model download and deployment
```
export VLLM_USE_MODELSCOPE=True
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m vllm.entrypoints.openai.api_server \
    --gpu-memory-utilization 0.95 \
    --served-model-name Lingma-SWE-GPT \
    --model yingwei/Lingma-SWE-GPT-v20240925\
    --tensor-parallel-size 4 \
    --max-model-len 131072 \
    --trust-remote-code \
    --rope-scaling '{"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}'

# test for deployment success
conda activate swesyninfer
python scripts/2_call_vllm.py
```
### Now You can run SWE-GPT on SWE-bench
```
python scripts/run.py conf/vanilla-lite-swebench.conf -f
```
### Evaluation on SWE-bench
We recommend using SWE-bench docker directly for evaluation.
Refer to the [SWE-bench](https://github.com/princeton-nlp/SWE-bench) repository for more details.

#### Note: we have built-in testbed examples. If you want to download the complete testbed, please download it from [here](https://modelscope.cn/datasets/Lingma/testbed/summary) and replace the testbed folder.



## TODO
- [x] upload 72B model
- [] upload 7B model
- [] upload technical report 

## Citation

## Acknowledgments

We would like to thank the [Qwen](https://github.com/QwenLM/Qwen2.5) team for their foundational work, which has been instrumental in the development of Lingma SWE-GPT.

We would also like to thank the [SWE-bench](https://github.com/princeton-nlp/SWE-bench), [AutoCodeRover](https://github.com/nus-apr/auto-code-rover), and [Agentless](https://github.com/OpenAutoCoder/Agentless) teams for their foundational work, which played an important role in the development of SWESynInfer.

