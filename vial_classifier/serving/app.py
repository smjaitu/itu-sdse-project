import atexit
import os
import subprocess
import sys
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Response

from vial_classifier.config import EXPORT_DIR, PORT, UPSTREAM_HOST, UPSTREAM_PORT

# The exported model to serve. Can be overridden with an environment variable (e.g. in Docker)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", EXPORT_DIR))

scoring_proc = None


def _start_scoring_server():
    env = os.environ.copy()
    # The scoring server execs `uvicorn` through PATH; make sure the
    # interpreter's bin directory is on it
    interpreter_bin = os.path.dirname(os.path.abspath(sys.executable))
    env["PATH"] = os.pathsep.join([interpreter_bin, env.get("PATH", "")])
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "mlflow", "models", "serve",
            "-m", str(MODEL_PATH),
            "-p", str(UPSTREAM_PORT),
            "--workers", "1",
            "--host", UPSTREAM_HOST,
            "--env-manager", "local",
        ],
        env=env,
    )
    atexit.register(lambda: proc.poll() is None and proc.terminate())
    return proc


@asynccontextmanager
async def lifespan(app):
    """Start the scoring server when the API starts, stop it when the API stops."""
    global scoring_proc
    scoring_proc = _start_scoring_server()
    print("Scoring server starting on port", UPSTREAM_PORT)
    yield
    if scoring_proc.poll() is None:
        scoring_proc.terminate()


app = FastAPI(docs_url=None, openapi_url=None, redoc_url=None, lifespan=lifespan)


def _upstream_health():
    """Return the scoring server's /health status code, None if unreachable."""
    try:
        with urllib.request.urlopen(
            f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}/health",
            timeout=1.0,
        ) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


@app.get("/health")
def health():
    if scoring_proc is not None and scoring_proc.poll() is not None:
        return Response(
            content='{"status": "error"}', status_code=503, media_type="application/json"
        )
    if _upstream_health() == 200:
        return Response(content='{"status": "ok"}', media_type="application/json")
    return Response(
        content='{"status": "loading"}', status_code=503, media_type="application/json"
    )


# ---- CLI ----
def main():
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()