# CommerceCore inference image -- runs identically on RunPod or a local
# machine (docker run), per EXECUTION_PLAN.md section 16b.
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    transformers peft bitsandbytes accelerate huggingface_hub \
    fastapi uvicorn pydantic

COPY serve/ /app/serve/

ENV COMMERCECORE_ADAPTER_PATH=""

EXPOSE 8000

CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
