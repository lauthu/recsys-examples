FROM vllm/vllm-openai@sha256:953d3a06d5e64ab582985cd7401289d3abf2a2c14ef2158e9a84313daeec77d7
# Build context: the clean vllm-project/je checkout pinned in the solution YAML.
COPY . /opt/vllm-gr
RUN python3 -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless && \
    python3 -m pip install --no-cache-dir /opt/vllm-gr
# vLLM 0.22.1 casts CUDA_VISIBLE_DEVICES to int for NVML queries. Resolve UUIDs
# through NVML instead, preserving the harness's exact CUDA device selection.
# Apply in the installed package so spawned GPU workers receive the same fix.
RUN python3 - <<'PY'
from importlib.util import find_spec
from pathlib import Path
path = Path(find_spec("vllm").origin).parent / "platforms/interface.py"
old = "            return int(physical_device_id)"
new = """            if physical_device_id.startswith("GPU-"):
                from vllm.utils.import_utils import import_pynvml
                nvml = import_pynvml()
                nvml.nvmlInit()
                try:
                    handle = nvml.nvmlDeviceGetHandleByUUID(physical_device_id)
                    return nvml.nvmlDeviceGetIndex(handle)
                finally:
                    nvml.nvmlShutdown()
            return int(physical_device_id)"""
text = path.read_text()
assert text.count(old) == 1, "vLLM device mapping changed; review UUID patch"
path.write_text(text.replace(old, new))
PY
ENV VLLM_USE_V2_MODEL_RUNNER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
