
export TMPDIR=/extra_data/vllm-berag/.tmp
export TEMP=/extra_data/vllm-berag/.tmp
export TMP=/extra_data/vllm-berag/.tmp

export UV_CACHE_DIR=/extra_data/vllm-berag/.cache/uv
export UV_PYTHON_INSTALL_DIR=/extra_data/vllm-berag/.uv-python
export PIP_CACHE_DIR=/extra_data/vllm-berag/.cache/pip
export XDG_CACHE_HOME=/extra_data/vllm-berag/.cache
export TORCH_HOME=/extra_data/vllm-berag/.cache/torch
export HF_HOME=/extra_data/vllm-berag/.cache/huggingface
export HF_HUB_CACHE=/extra_data/vllm-berag/.cache/huggingface/hub
export TRITON_CACHE_DIR=/extra_data/vllm-berag/.cache/triton

export PATH=/extra_data/vllm-berag/.local/bin:$PATH

source .venv/bin/activate
