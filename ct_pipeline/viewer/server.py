from __future__ import annotations

import re
import tempfile
import uuid
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image

from .session import ViewerSession, load_viewer_session


class ViewerSessionStore:
    def __init__(self, session: ViewerSession | None = None) -> None:
        self._session = session
        self._lock = Lock()

    def get(self) -> ViewerSession | None:
        with self._lock:
            return self._session

    def set(self, session: ViewerSession) -> None:
        with self._lock:
            self._session = session


def _encode_grayscale_png(image: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    uint8_image = np.round(clipped * 255.0).astype(np.uint8)
    pil_image = Image.fromarray(uint8_image, mode="L")
    from io import BytesIO

    handle = BytesIO()
    pil_image.save(handle, format="PNG")
    return handle.getvalue()


def _frontend_dist_path() -> Path:
    return Path(__file__).resolve().parents[2] / "viewer" / "dist"


def _safe_upload_filename(filename: str) -> str:
    basename = Path(filename or "selected.ply").name
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", basename).strip("._")
    if not cleaned:
        cleaned = "selected.ply"
    if not cleaned.lower().endswith(".ply"):
        cleaned = f"{cleaned}.ply"
    return cleaned[:120]


async def _save_uploaded_ply(request, filename: str) -> Path:
    upload_dir = Path(tempfile.gettempdir()) / "ctgs_viewer_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"{uuid.uuid4().hex}_{_safe_upload_filename(filename)}"

    bytes_written = 0
    with target.open("wb") as handle:
        async for chunk in request.stream():
            if not chunk:
                continue
            bytes_written += len(chunk)
            handle.write(chunk)

    if bytes_written <= 0:
        target.unlink(missing_ok=True)
        raise ValueError("Uploaded PLY is empty.")
    return target


def create_viewer_app(session: ViewerSession | None = None, *, load_device=None):
    try:
        from fastapi import FastAPI, HTTPException, Query, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse, Response
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("FastAPI is required for the CTGS viewer. Install fastapi and uvicorn first.") from exc

    globals()["Request"] = Request
    session_store = ViewerSessionStore(session)
    app = FastAPI(title="CTGS Viewer", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def require_session() -> ViewerSession:
        current_session = session_store.get()
        if current_session is None:
            raise HTTPException(status_code=404, detail="No PLY session is loaded.")
        return current_session

    @app.get("/api/session")
    def get_session():
        return require_session().session_payload()

    @app.post("/api/session/load")
    async def load_session_from_upload(request: Request, filename: str = Query("selected.ply")):
        try:
            upload_path = await _save_uploaded_ply(request, filename)
            loaded_session = load_viewer_session(upload_path, device=load_device)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to load PLY: {exc}") from exc

        session_store.set(loaded_session)
        return loaded_session.session_payload()

    @app.get("/api/gaussians/meta")
    def get_gaussian_meta():
        return require_session().gaussian_meta_payload()

    @app.get("/api/gaussians/buffer")
    def get_gaussian_buffer():
        current_session = require_session()
        return Response(content=current_session.gaussian_buffer, media_type="application/octet-stream")

    @app.get("/api/slice/gs")
    def get_generated_slice(
        axis: str = Query(..., pattern="^(x|y|z)$"),
        t: float = Query(0.5, ge=0.0, le=1.0),
        layer: str = Query("all", pattern="^(all|surface|bulk)$"),
        size: int = Query(512, ge=64, le=1024),
        intensity_clip: bool = Query(False),
        intensity_min: float = Query(0.0, ge=0.0, le=1.0),
        intensity_max: float = Query(1.0, ge=0.0, le=1.0),
        preview: str = Query("raw", pattern="^(raw|clipped|mask)$"),
    ):
        try:
            image = require_session().render_slice(
                axis=axis,
                t=t,
                layer=layer,
                size=size,
                intensity_min=intensity_min,
                intensity_max=intensity_max,
                intensity_clip=intensity_clip,
                preview_mode=preview,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=_encode_grayscale_png(image), media_type="image/png")

    @app.get("/healthz")
    def healthz():
        current_session = session_store.get()
        return {
            "status": "ok",
            "loaded": current_session is not None,
            "ply": str(current_session.ply_path) if current_session is not None else None,
        }

    frontend_dist = session.frontend_dist if session is not None else _frontend_dist_path()
    if frontend_dist is not None and frontend_dist.exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="viewer-assets")

        @app.get("/", include_in_schema=False)
        def get_index():
            return HTMLResponse((frontend_dist / "index.html").read_text(encoding="utf-8"))

    else:
        @app.get("/", include_in_schema=False)
        def get_index():
            return HTMLResponse(
                "<html><body><h1>CTGS Viewer API</h1><p>Frontend assets are missing. Run npm install && npm run build in viewer/.</p></body></html>"
            )

    return app
