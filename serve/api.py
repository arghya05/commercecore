"""Thin REST API wrapper around CommerceCoreQueryParser.

Run with: uvicorn serve.api:app --host 0.0.0.0 --port 8000
Requires: pip install fastapi uvicorn (in addition to the training/inference deps)

This is the concrete "startup integrates in 5 minutes" interface --
POST a query, get structured constraints back. Runs identically on a
RunPod pod or a local machine (same Docker image, per EXECUTION_PLAN.md
section 16b's "one container image, two deployment targets" decision).
"""

import os

from fastapi import FastAPI
from pydantic import BaseModel

from serve.inference import CommerceCoreQueryParser, BASE_MODEL_ID

ADAPTER_PATH = os.environ.get("COMMERCECORE_ADAPTER_PATH") or None  # set if serving a raw, unmerged adapter
BASE_MODEL_PATH = os.environ.get("COMMERCECORE_BASE_MODEL_PATH", BASE_MODEL_ID)  # merged model dir, or HF id

app = FastAPI(title="CommerceCore Query Parser", version="0.1.0")
_parser = None


class ParseRequest(BaseModel):
    query: str


class ParseResponse(BaseModel):
    query: str
    constraints: list


@app.on_event("startup")
def load_model():
    global _parser
    _parser = CommerceCoreQueryParser(base_model_path=BASE_MODEL_PATH, adapter_path=ADAPTER_PATH)


@app.post("/parse-query", response_model=ParseResponse)
def parse_query(req: ParseRequest):
    constraints = _parser.parse(req.query)
    return ParseResponse(query=req.query, constraints=constraints)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _parser is not None}
