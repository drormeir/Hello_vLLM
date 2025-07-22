# Hello World vLLM

## TL;DR
The [notebook](./Hello_World_vLLM.ipynb) is a concise, quick-start guide that sets up vLLM, loads a 125M model, demonstrates the generation of replies, and showcases the manipulation of the model's log probabilities and tokens. It is a minimal template for experiments with larger models, streaming, and benchmarking.

## High-level flow:
1. Environment bootstrap
2. LLM instantiation: LLM(model= "facebook/opt-125m") creates a small model that fits on almost any GPU.
3. Prompt-to-Text demo: builds a list of prompts, sets sampling parameters, and generates a set of replies for each prompt, in three different use cases.
4. Find the top three generated texts based on their tokens' average log probability.

## Prompt-to-Text use cases:
This notebook shows three basic use cases of generating text replies from prompts:
Single prompt to single reply
Single prompt to many replies
Many prompts to a single reply (each of the prompts)

## Top three replies:
The last cell of this notebook contains a program that generates 10 different replies for a single prompt. Then, it calculates the average log probability for each reply based on the probabilities of its selected tokens.<br>
It sorts the outputs according to the calculation result and displays the top three.

## Possible simple extensions for this notebook:
1. Load a larger model, such as meta-llama/Llama-3-8b if hardware allows.
2. Wrap 'llm.generate' in a FastAPI endpoint and load-test it.
3. Compare performance with Hugging Face TextGenerationInference.

Key references
Official docs at https://vllm.ai/
