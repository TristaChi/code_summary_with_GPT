# Transfer Attacks and Defenses for Large Language Models on Coding Tasks

This repository contains the dataset and code used in our paper on evaluating the robustness of Large Language Models (LLMs) against adversarially perturbed inputs. It provides a framework for running experiments to test LLMs' results on code summarization tasks to both original and perturbed inputs. 

## Usage

1. Clone the repository.
2. Install the `openAI` package using the following command:
```bash
pip install openai==0.28
```

## Repository Structure

The repository is organized into several key directories and files:

### `data/`

This directory holds the dataset of 1,000 data points used in our study. Each data point consists of three components:

- **Original Input** 
- **Adversarially Perturbed Input** 
- **Target** 

### `results/`

This file contains the results of each LLM. The results of Meta-Prompts are in files meta_p_gpt4.csv and meta_p_gpt35.csv. Each LLM's other results are in a file of its own name. 

### `run.py`

The entry point of running the whole experiment thoroughly. Run with `python3 run.py --file_name={filename} --data_dir={datadir} --use_GPT={the LLM to use}`
Models should be one of the following:`{"gpt35", "gpt4", "codeLlama", "Claude1", "Claude2"}`

### `run_GPT.py`
The main file of the experiment. 

### `eval.py`
The file to evaluate LLM results. 

### `prompt.py`
The file that saves all the prompts used for the experiment. 

