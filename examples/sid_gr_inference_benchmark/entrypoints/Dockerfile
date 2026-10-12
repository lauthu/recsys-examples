# Extend an existing SGLang environment without replacing its framework stack.
# The selected base must provide Python, uv, native beam search and GR dependencies.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG CUDA_COMPILER_VERSION=13.0.88
RUN uv pip install --python "$(command -v python3)" \
    "nvidia-cuda-nvcc==${CUDA_COMPILER_VERSION}" \
    "nvidia-cuda-crt==${CUDA_COMPILER_VERSION}" \
    "nvidia-nvvm==${CUDA_COMPILER_VERSION}"
