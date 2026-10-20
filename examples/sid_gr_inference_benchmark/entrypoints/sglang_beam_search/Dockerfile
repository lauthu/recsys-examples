# Build with https://github.com/cswuyg/sglang at
# af4e3f42a5a1a79ffc82d01a42f7e96f74294b8d as the build context:
# docker build -f entrypoints/sglang_beam_search/Dockerfile \
#   -t sid-gr-benchmark:beam-af4e3f4-cu130 "$SGLANG_BEAM_SOURCE"
FROM lmsysorg/sglang@sha256:b5c488694d74c13587af6057b90220498c4869571c8e1e3c43dd6f43ae721306
RUN uv pip install --python /opt/sglang/bin/python \
    --index-url https://download.pytorch.org/whl/cu130 \
    'torch==2.9.1' 'torchvision==0.24.1' 'torchaudio==2.9.1'
RUN uv pip install --python /opt/sglang/bin/python --no-deps \
    --index-url https://docs.sglang.ai/whl/cu130/ 'sglang-kernel==0.4.1.post1'
COPY python/pyproject.toml /tmp/beam-pyproject.toml
RUN python3 -c "import tomllib; print('\n'.join(tomllib.load(open('/tmp/beam-pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/beam-requirements.txt \
    && uv pip install --python /opt/sglang/bin/python -r /tmp/beam-requirements.txt
# Discard the base image's optional cache from a different FlashInfer version.
# The pinned FlashInfer builds its own kernels during untimed setup/warmup.
RUN uv pip uninstall --python /opt/sglang/bin/python flashinfer-jit-cache
# Both candidates use one runtime; these versions support the external GR kernel.
RUN uv pip install --python /opt/sglang/bin/python \
    'nvidia-cutlass-dsl[cu13]==4.5.1' 'quack-kernels==0.4.1' 'torch-c-dlpack-ext==0.1.5'
# Uninstall leaves generated cache files that Python treats as a namespace package.
RUN rm -rf /opt/sglang/lib/python3.12/site-packages/flashinfer_jit_cache
