FROM --platform=linux/amd64 python:3.13-slim-bookworm@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26
LABEL qfbench2.interface_version="2.0" \
      org.opencontainers.image.title="Under25 AgentHon Track 4 Final" \
      org.opencontainers.image.description="Current Under25 forecasting workflow with House-only final submission adapter"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/under25 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    XDG_CACHE_HOME=/tmp/cache OMP_NUM_THREADS=4
WORKDIR /opt/under25
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt && python -m pip check
COPY workflow/ ./workflow/
COPY scripts/ ./scripts/
COPY prompts/ ./prompts/
COPY profiles/ ./profiles/
COPY vendor/ ./vendor/
COPY data/*.json ./data/
COPY data/query-term-ac-index.json.gz ./data/
COPY submission_entry.py submission_runtime.py ./
RUN python -c "import submission_entry; from submission_runtime import SubmissionWorkflow"
WORKDIR /output
ENTRYPOINT ["python", "/opt/under25/submission_entry.py"]
