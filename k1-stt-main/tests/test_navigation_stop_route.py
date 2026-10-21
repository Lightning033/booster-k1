"""Exercise the real STT HTTP handler and VLN client using in-memory transport."""

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from stt.config import Config
from stt.server import ConsoleHandler, ConsoleState


class MemoryConnection:
    """Socket substitute for BaseHTTPRequestHandler; no listener or robot needed."""

    def __init__(self, method, path):
        self.request = io.BytesIO(
            f"{method} {path} HTTP/1.0\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n".encode())
        self.response = io.BytesIO()

    def makefile(self, mode, buffering=-1):
        return self.request

    def sendall(self, data):
        self.response.write(data)


def request_console(console, method, path):
    connection = MemoryConnection(method, path)
    ConsoleHandler(connection, ("127.0.0.1", 12345), SimpleNamespace(console=console))
    headers, body = connection.response.getvalue().split(b"\r\n\r\n", 1)
    status = int(headers.split(b" ", 2)[1])
    return status, body


class NavigationStopRouteTests(unittest.TestCase):
    def setUp(self):
        config = Config()
        config.vln.enabled = True
        self.pipeline = Mock(cfg=config)
        self.pipeline.transcriber.device = "cpu"
        self.pipeline.transcriber.compute_type = "int8"
        self.pipeline.recorder.level = 0.0
        self.console = ConsoleState(self.pipeline)
        self.console.state = "executing"
        self.console.task = self.console._new_task("voice-1", "Find the chair", "running")

    def test_stop_route_forwards_request_without_unlocking_listening(self):
        reply = {"status": "stopping", "command_id": "voice-1"}
        with patch("stt.vln_client.urllib.request.urlopen",
                   return_value=io.BytesIO(json.dumps(reply).encode())) as urlopen:
            status, body = request_console(self.console, "POST", "/api/navigation/stop")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), reply)
        urlopen.assert_called_once()
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/stop")
        self.assertEqual(request.get_method(), "POST")
        self.assertIsNone(request.data)
        self.assertEqual(self.console.state, "executing")
        self.assertIsNone(self.console.task["ended"])
        with self.assertRaises(RuntimeError):
            self.console.start()
        self.pipeline.stop_listening.assert_not_called()

    def test_failed_request_returns_502_preserves_task_and_allows_retry(self):
        errors = (URLError("Connection refused"), TimeoutError("Timed out"),
                  HTTPError("http://127.0.0.1:8000/stop", 503, "Unavailable", {}, None))
        task_before = dict(self.console.task)
        for error in errors:
            with self.subTest(error=type(error).__name__):
                with patch("stt.vln_client.urllib.request.urlopen", side_effect=error) as urlopen:
                    status, body = request_console(self.console, "POST", "/api/navigation/stop")
                self.assertEqual(status, 502)
                self.assertIn("Could not confirm navigation stop", json.loads(body)["error"])
                urlopen.assert_called_once()
                self.assertEqual(self.console.state, "executing")
                self.assertEqual(self.console.task, task_before)
        with patch("stt.vln_client.urllib.request.urlopen",
                   return_value=io.BytesIO(b'{"status":"stopping","command_id":"voice-1"}')):
            self.assertEqual(request_console(self.console, "POST", "/api/navigation/stop")[0], 200)

    def test_invalid_reply_returns_502_without_claiming_stop_succeeded(self):
        for reply in (b"not json", b"{}", b"null", b'{"status":"running"}'):
            with self.subTest(reply=reply):
                with patch("stt.vln_client.urllib.request.urlopen", return_value=io.BytesIO(reply)):
                    status, body = request_console(self.console, "POST", "/api/navigation/stop")
                self.assertEqual(status, 502)
                self.assertIn("error", json.loads(body))
                self.assertEqual(self.console.state, "executing")

    def test_disabled_integration_returns_409_without_contacting_vln(self):
        self.console.vln = None
        with patch("stt.vln_client.urllib.request.urlopen") as urlopen:
            status, body = request_console(self.console, "POST", "/api/navigation/stop")
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(body), {"error": "VLN integration is disabled"})
        urlopen.assert_not_called()

    def test_terminal_and_idle_replies_are_forwarded_for_polling_to_reconcile(self):
        for state in ("idle", "completed", "cancelled", "failed"):
            with self.subTest(state=state):
                reply = {"status": state, "command_id": None if state == "idle" else "voice-1"}
                with patch("stt.vln_client.urllib.request.urlopen",
                           return_value=io.BytesIO(json.dumps(reply).encode())):
                    status, body = request_console(self.console, "POST", "/api/navigation/stop")
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body), reply)

    def test_stop_listening_still_uses_the_original_route(self):
        self.console.state = "listening"
        result = {"accepted": False, "transcript": ""}
        self.pipeline.stop_listening.return_value.to_dict.return_value = result
        with patch("stt.vln_client.urllib.request.urlopen") as urlopen:
            status, body = request_console(self.console, "POST", "/api/stop")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), result)
        self.pipeline.stop_listening.assert_called_once_with()
        urlopen.assert_not_called()
        self.assertEqual(self.console.state, "idle")

    def test_existing_get_and_start_routes_work_with_top_level_handler(self):
        for path in ("/", "/api/status", "/api/level", "/api/commands", "/api/latest"):
            with self.subTest(path=path):
                self.assertEqual(request_console(self.console, "GET", path)[0], 200)
        self.console.state = "idle"
        self.assertEqual(request_console(self.console, "POST", "/api/start")[0], 200)
        self.pipeline.start_listening.assert_called_once_with()
        self.assertEqual(self.console.state, "listening")

    def test_get_cannot_request_navigation_stop(self):
        with patch("stt.vln_client.urllib.request.urlopen") as urlopen:
            self.assertEqual(request_console(self.console, "GET", "/api/navigation/stop")[0], 404)
        urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
