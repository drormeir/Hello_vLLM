# Hello_vLLM
This repo is a personal experiment to demonstrate basic vLLM functionality.<br>
I welcome ideas, pull requests, and connections on [LinkedIn](https://www.linkedin.com/in/dror-meirovich/).

## Make vLLM work on a Paperspace A5000
I need the editable install version of vLLM (pip install -e .) because I plan to add my own CUDA kernels to vLLM's source.<br>
But the vanilla `pip install -e .` fetches the latest PyTorch wheel, which is built for CUDA 12.x. On a Paperspace A5000, this collides with the pre-installed CUDA 12.0 toolkit and breaks the vLLM build.

The fix it, I will stick to the stable CUDA 11.8 toolchain:
1. Install NVIDIA Toolkit 11.8 silently (so nvcc and headers match).
2. Pin PyTorch 2.7.0 + cu118 before touching vLLM.
3. Run `pip install -e .` inside the local vLLM clone.
vLLM now finds a matching toolkit + wheel and compiles its CUDA extensions. It will be ready for custom kernel development, while the system's existing 12.x runtime stays intact for other work.

### Install CUDA 11.8 toolkit
<pre><code>
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit --override
</code></pre>

### Create an empty venv
<pre><code>
python3 -m venv ~/vllm_venv
source ~/vllm_venv/bin/activate
python -m pip install --upgrade pip
</code></pre>

### Pin PyTorch to the CUDA 11.8 wheel
<pre><code>
  python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118
</code></pre>

### Clone vLLM and build it
<pre><code>
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
</code></pre>

### Extras for notebooks
<pre><code>
pip install datasets sentence-transformers matplotlib notebook
python -m ipykernel install --name vllm_venv \
    --display-name "Python (vllm_venv)" --prefix=/usr/local
</code></pre>



### Smoke test
<pre><code>
nvcc
python - <<'PY'
import torch, subprocess, os
print("Torch CUDA version:", torch.version.cuda)      # ➜ 11.8
subprocess.run(["nvcc", "--version"])                 # ➜ release 11.8
PY
</code></pre>

It should print `11.8` for both the Torch and `nvcc`.

### Proceed to run the notebooks
Then open `Hello_World_vLLM.ipynb` with kernel `Python (vllm_venv)`. It should generate a sample answer without errors.


## First notebook: Hello_World_vLLM
The [first notebook](./Hello_World_vLLM.ipynb) is a concise, quick-start guide that sets up vLLM, loads a 125M model, demonstrates the generation of replies, and showcases the manipulation of the model's log probabilities and tokens. It is a minimal template for experiments with larger models, streaming, and benchmarking.

### High-level flow:
1. Environment bootstrap
2. LLM instantiation: LLM(model= "facebook/opt-125m") creates a small model that fits on almost any GPU.
3. Prompt-to-Text demo: builds a list of prompts, sets sampling parameters, and generates a set of replies for each prompt, in three different use cases.
4. Find the top three generated texts based on their tokens' average log probability.

### Prompt-to-Text use cases:
This notebook shows three basic use cases of generating text replies from prompts:
Single prompt to single reply
Single prompt to many replies
Many prompts to a single reply (each of the prompts)

### Top three replies:
The last cell of this notebook contains a program that generates 10 different replies for a single prompt. Then, it calculates the average log probability for each reply based on the probabilities of its selected tokens.<br>
It sorts the outputs according to the calculation result and displays the top three.

## Second notebook: Draft Rerank Analysis

In this [second notebook](./Draft_Rerank_Analysis.ipynb), I utilize vLLM to conduct a large-scale experiment on a language model's probability estimates.<br>
First, the code bootstraps the environment end-to-end: it detects the GPU and CUDA driver, then loads an open-source 125-million-parameter LLM along with a sentence-embedding model.<br>
Next, I import a real conversation dataset (Open-Assistant) and turn it into 250 question-answer pairs. For every question the notebook asks the LLM to sample 100 alternative answers in one pass, generating tens of thousands of responses while keeping memory and runtime under tight control.<br>
The same model then scores its answers. I compare those scores against two intuitive quality signals:<br>
1. Word-overlap rate with the reference answer.
2. Semantic similarity in embedding space.

By computing Spearman correlations, I test whether the model's built-in "confidence" really tracks answer quality.<br>
Finally, I visualise the findings in two concise histograms. The takeaway is that, for most prompts, simply sampling multiple times and selecting the response with the highest average log-probability yields noticeably better answers, even with a small model and no additional training.

## Official docs at
https://vllm.ai/
