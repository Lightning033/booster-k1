"""HTTP entry point for STT. Run one server process, with one worker and no reload.

POST /command accepts {"command_id": "...", "text": "..."} immediately.
POST /stop requests interruption; repeated requests do not interrupt cleanup.
GET /status reports idle, running, stopping, completed, cancelled, or failed.
Normal shutdown waits for the current navigation run, including its cleanup.
Server shutdown does not yet request cancellation. This is not an emergency stop.
"""

import asyncio
import json
import logging
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, StringConstraints

LOGGER = logging.getLogger("uvicorn.error")
ROOT = Path(__file__).resolve().parent
NonEmptyText = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]


class Command(BaseModel):
    text: NonEmptyText
    command_id: Optional[NonEmptyText] = None


class NavigationProcessError(RuntimeError):
    """A child process failure whose message is already suitable for STT."""


class NavigationCancelled(Exception):
    """Navigation ended after an operator's stop request."""


class NavigationProcess:
    """Run the pipeline in a fresh interpreter and wait for its final result."""

    def __init__(self, worker_path: Path = ROOT / "vln_worker.py"):
        self._worker_path = worker_path.resolve()

    def __call__(self, text: str, stop_requested: Optional[threading.Event] = None) -> None:
        if stop_requested is not None and stop_requested.is_set():
            raise NavigationCancelled("Navigation stopped before starting.")
        with tempfile.TemporaryDirectory(prefix="k1-vln-") as directory:
            result_path = Path(directory) / "result.json"
            # A separate session prevents terminal interrupts reaching the child
            # directly. Only the monitor sends SIGINT, once the worker is ready.
            with subprocess.Popen(
                [sys.executable, "-u", str(self._worker_path),
                 "--result", str(result_path), "--", text],
                cwd=ROOT, start_new_session=True,
            ) as process:
                interrupt_sent = False
                while True:
                    if (stop_requested is not None and stop_requested.is_set()
                            and not interrupt_sent and result_path.with_suffix(".ready").exists()):
                        process.send_signal(signal.SIGINT)
                        interrupt_sent = True
                    try:
                        returncode = process.wait(timeout=0.05)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            self._check_result(result_path, returncode)

    @staticmethod
    def _check_result(result_path: Path, returncode: int) -> None:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise NavigationProcessError(
                f"Navigation process exited with code {returncode} without a valid result. "
                "See the VLN terminal for details.") from exc
        if isinstance(result, dict):
            if returncode == 0 and result.get("status") == "cancelled":
                raise NavigationCancelled(result.get("message", "Navigation stopped by operator."))
            if returncode == 0 and result.get("status") == "completed":
                return
            if result.get("status") == "failed" and isinstance(result.get("message"), str):
                raise NavigationProcessError(result["message"])
        raise NavigationProcessError(
            f"Navigation process returned an inconsistent result (exit code {returncode}).")


class NavigationRunner:
    """Own one active run and its status; never queue a second instruction."""

    def __init__(self, navigate: Callable[[str], None]):
        self._navigate = navigate
        self._lock = threading.Lock()
        self._thread = None
        self._closing = False
        self._started = None
        self._finished = None
        self._stop_requested = threading.Event()
        self._state = {"status": "idle", "command_id": None, "text": None, "message": ""}

    def submit(self, cmd: Command) -> dict:
        with self._lock:
            if self._closing:
                raise HTTPException(status_code=503, detail="VLN server is shutting down")
            if self._state["status"] in ("running", "stopping"):
                return {"status": "busy", "command_id": self._state["command_id"]}
            command_id = cmd.command_id or uuid.uuid4().hex
            self._state.update(status="running", command_id=command_id,
                               text=cmd.text, message="")
            self._started = time.monotonic()
            self._finished = None
            self._stop_requested = threading.Event()
            worker = threading.Thread(target=self._run, args=(command_id, cmd.text),
                                      name="vln_navigation", daemon=False)
            try:
                worker.start()
            except Exception as exc:
                self._state.update(status="failed", message=f"Could not start navigation: {exc}")
                self._finished = time.monotonic()
                raise HTTPException(status_code=503, detail=self._state["message"]) from exc
            self._thread = worker
        return {"status": "accepted", "command_id": command_id}

    def _run(self, command_id: str, text: str) -> None:
        status, message = "completed", ""
        try:
            LOGGER.info("VLN running %s: %s", command_id, text)
            if isinstance(self._navigate, NavigationProcess):
                self._navigate(text, self._stop_requested)
            else:
                self._navigate(text)
        except NavigationCancelled as exc:
            status, message = "cancelled", str(exc)
        except NavigationProcessError as exc:
            status, message = "failed", str(exc)
            LOGGER.error("VLN failed %s: %s", command_id, message)
        except BaseException as exc:
            # Even a worker's SystemExit/KeyboardInterrupt must release the busy state.
            status, message = "failed", f"{type(exc).__name__}: {exc}"
            LOGGER.exception("VLN failed %s", command_id)
        finally:
            with self._lock:
                self._state.update(status=status, message=message)
                self._finished = time.monotonic()
        LOGGER.info("VLN %s: %s", status, command_id)

    def stop(self) -> dict:
        """Request one interrupt; the monitor owns signal delivery and reaping."""
        with self._lock:
            if self._state["status"] == "running":
                if not isinstance(self._navigate, NavigationProcess):
                    raise HTTPException(status_code=501, detail="Injected navigation does not support stopping")
                self._state.update(status="stopping", message="Stop requested; waiting for navigation cleanup.")
                self._stop_requested.set()
            return {"status": self._state["status"], "command_id": self._state["command_id"]}

    def status(self) -> dict:
        with self._lock:
            elapsed = 0.0
            if self._started is not None:
                end = self._finished if self._finished is not None else time.monotonic()
                elapsed = round(end - self._started, 1)
            return {**self._state, "elapsed_s": elapsed}

    def close(self) -> None:
        """Reject further commands and wait for navigation to finish its cleanup."""
        with self._lock:
            self._closing = True
            worker = self._thread
        if worker is not None:
            worker.join()  # Never hold the state lock while waiting for the worker.


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        await asyncio.to_thread(app.state.runner.close)


def command(cmd: Command, request: Request) -> dict:
    return request.app.state.runner.submit(cmd)


def status(request: Request) -> dict:
    return request.app.state.runner.status()


def stop(request: Request) -> dict:
    return request.app.state.runner.stop()


def create_app(navigate: Optional[Callable[[str], None]] = None) -> FastAPI:
    """Inject a fake navigation function for offline HTTP tests."""
    application = FastAPI(title="Booster K1 VLN", lifespan=lifespan)
    application.state.runner = NavigationRunner(navigate if navigate is not None else NavigationProcess())
    application.add_api_route("/command", command, methods=["POST"])
    application.add_api_route("/status", status, methods=["GET"])
    application.add_api_route("/stop", stop, methods=["POST"])
    return application


app = create_app()
