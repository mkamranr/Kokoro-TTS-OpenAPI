from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    service = getattr(request.app.state, "service", None)
    if service is None:
        return {"status": "loading", "model_loaded": False}
    info = service.info()
    return {"status": "ok", "model_loaded": True, **info}
