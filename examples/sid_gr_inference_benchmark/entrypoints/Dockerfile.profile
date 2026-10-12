# Optional: copy compatible Nsight installations from a development image.
ARG PROFILER_IMAGE
ARG BASE_IMAGE
FROM ${PROFILER_IMAGE} AS profiler
FROM ${BASE_IMAGE}
ARG NSYS_ROOT
ARG NCU_ROOT
COPY --from=profiler ${NSYS_ROOT} /opt/bench-nsys
COPY --from=profiler ${NCU_ROOT} /opt/bench-ncu
RUN ln -sf /opt/bench-nsys/bin/nsys /usr/local/bin/nsys && \
    ln -sf /opt/bench-ncu/ncu /usr/local/bin/ncu
