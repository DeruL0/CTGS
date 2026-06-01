from __future__ import annotations

import argparse
from pathlib import Path

from .server import create_viewer_app
from .session import load_viewer_session


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CTGS viewer service")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Serve the local CTGS viewer")
    serve_parser.add_argument("--ply", help="Optional path to a CTGS display PLY to load at startup.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    serve_parser.add_argument("--port", default=8000, type=int, help="Port to bind.")
    serve_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Tensor device for the loaded PLY.")
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "serve":
        parser.error(f"Unsupported command: {args.command}")

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("uvicorn is required for the CTGS viewer. Install fastapi and uvicorn first.") from exc

    load_device = None if args.device == "auto" else args.device
    session = load_viewer_session(Path(args.ply), device=load_device) if args.ply else None
    app = create_viewer_app(session, load_device=load_device)
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
