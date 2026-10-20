# Dependencies for the existing gr_decode_atten source used by this migration.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
RUN uv pip install --python "$(command -v python3)" \
    'nvidia-cutlass-dsl[cu13]==4.5.1' 'quack-kernels==0.4.1'
