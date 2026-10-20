# Optional: copy Nsight Systems from a development image.
ARG PROFILER_IMAGE
ARG BASE_IMAGE
FROM ${PROFILER_IMAGE} AS profiler
FROM ${BASE_IMAGE}
ARG NSYS_ROOT
COPY --from=profiler ${NSYS_ROOT} /opt/bench-nsys
RUN ln -sf /opt/bench-nsys/bin/nsys /usr/local/bin/nsys
