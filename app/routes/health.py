from fastapi import APIRouter, Request

import app as app_package

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    # `models_dir` is the cache huggingface_hub actually resolved, not a guess:
    # it makes the "weights live in the project" guarantee observable instead
    # of silent. See app/__init__.py.
    payload = {"models_dir": app_package.resolved_cache_dir()}
    service = getattr(request.app.state, "service", None)
    if service is None:
        return {"status": "loading", "model_loaded": False, **payload}
    return {"status": "ok", "model_loaded": True, **service.info(), **payload}
