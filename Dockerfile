FROM python:3.12-slim
ARG POLARS_VERSION
RUN pip install --no-cache-dir "${POLARS_VERSION}" numpy
WORKDIR /work
ENV POLARS_MAX_THREADS=4
CMD ["python", "-u", "experiment.py"]
