"""VLN client request and failure handling without a running robot server."""

import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from stt.vln_client import VlnClient


class VlnClientTests(unittest.TestCase):
    def test_stop_posts_to_configured_server_and_returns_current_state(self):
        client = VlnClient("http://127.0.0.1:8000/", request_timeout_s=2.5)
        for status in ("stopping", "idle", "completed", "cancelled", "failed"):
            with self.subTest(status=status):
                reply = {"status": status, "command_id": None if status == "idle" else "voice-1"}
                with patch("stt.vln_client.urllib.request.urlopen",
                           return_value=io.BytesIO(json.dumps(reply).encode("utf-8"))) as urlopen:
                    self.assertEqual(client.stop(), reply)
                urlopen.assert_called_once()
                request = urlopen.call_args.args[0]
                self.assertEqual(request.full_url, "http://127.0.0.1:8000/stop")
                self.assertEqual(request.get_method(), "POST")
                self.assertIsNone(request.data)
                self.assertEqual(urlopen.call_args.kwargs, {"timeout": 2.5})

    def test_stop_propagates_errors_without_retrying_or_claiming_success(self):
        client = VlnClient("http://127.0.0.1:8000")
        errors = (
            URLError("Connection refused"),
            TimeoutError("Request timed out"),
            HTTPError(client.base_url + "/stop", 503, "Unavailable", {}, None),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__):
                with patch("stt.vln_client.urllib.request.urlopen", side_effect=error) as urlopen:
                    with self.assertRaises(type(error)) as raised:
                        client.stop()
                self.assertIs(raised.exception, error)
                urlopen.assert_called_once()

    def test_invalid_stop_response_is_not_reported_as_success(self):
        client = VlnClient("http://127.0.0.1:8000")
        with patch("stt.vln_client.urllib.request.urlopen", return_value=io.BytesIO(b"not json")):
            with self.assertRaises(json.JSONDecodeError):
                client.stop()


if __name__ == "__main__":
    unittest.main()
