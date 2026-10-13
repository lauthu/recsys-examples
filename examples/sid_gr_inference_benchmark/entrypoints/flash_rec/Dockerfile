FROM vllm/vllm-openai@sha256:953d3a06d5e64ab582985cd7401289d3abf2a2c14ef2158e9a84313daeec77d7
# Build context: the FlashRec checkout pinned in the solution YAML.
COPY . /opt/FlashRec
RUN python3 -m pip install --no-cache-dir /opt/FlashRec
