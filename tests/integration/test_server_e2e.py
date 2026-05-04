"""
End-to-end integration tests for the Mugen HTTP server.

Requires mugen-server binary to be built first.
Run:  python -m unittest tests/integration/test_server_e2e.py
"""

import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
import unittest


def _find_server_binary():
    """Search common build locations for the mugen-server binary."""
    candidates = [
        os.path.join("build", "src", "mugen-server"),
        os.path.join("build", "mugen-server"),
        os.path.join("cmake-build-debug", "src", "mugen-server"),
        os.path.join("cmake-build-release", "src", "mugen-server"),
    ]
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for rel in candidates:
        path = os.path.join(project_root, rel)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    env_path = os.environ.get("MUGEN_SERVER_BIN")
    if env_path and os.path.isfile(env_path):
        return env_path
    return None


def _find_free_port():
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(host, port, timeout=10.0):
    """Block until the given TCP port accepts connections or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


SERVER_BIN = _find_server_binary()


@unittest.skipIf(SERVER_BIN is None, "mugen-server binary not found — skipping E2E tests")
class TestMugenServerE2E(unittest.TestCase):
    """Integration tests that start a real mugen-server and talk HTTP to it."""

    host = "127.0.0.1"
    port = 0
    process = None

    @classmethod
    def setUpClass(cls):
        cls.port = _find_free_port()
        cmd = [SERVER_BIN, "--host", cls.host, "--port", str(cls.port)]

        model_path = os.environ.get("MUGEN_TEST_MODEL")
        if model_path:
            cmd.extend(["--model", model_path])

        cls.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if not _wait_for_port(cls.host, cls.port, timeout=15.0):
            cls.process.kill()
            cls.process.wait()
            raise RuntimeError(f"mugen-server did not start on {cls.host}:{cls.port}")

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            cls.process.send_signal(signal.SIGTERM)
            try:
                cls.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.process.kill()
                cls.process.wait()

    def _conn(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=5)

    # ----- /health -----

    def test_health_endpoint(self):
        """GET /health should return 200."""
        conn = self._conn()
        conn.request("GET", "/health")
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)
        body = json.loads(resp.read())
        self.assertIn("status", body)
        conn.close()

    # ----- /v1/models -----

    def test_models_endpoint(self):
        """GET /v1/models should return 200 with an object/data key."""
        conn = self._conn()
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        self.assertIn(resp.status, (200, 404))
        if resp.status == 200:
            body = json.loads(resp.read())
            self.assertIn("data", body)
        conn.close()

    # ----- /v1/chat/completions -----

    @unittest.skipUnless(
        os.environ.get("MUGEN_TEST_MODEL"),
        "No model loaded — skipping chat completion test",
    )
    def test_chat_completions(self):
        """POST /v1/chat/completions with a simple prompt."""
        conn = self._conn()
        payload = json.dumps({
            "model": "test",
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 16,
            "stream": False,
        })
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)
        body = json.loads(resp.read())
        self.assertIn("choices", body)
        conn.close()

    # ----- unknown route → 404 -----

    def test_unknown_route_returns_404(self):
        """GET on a non-existent path should return 404."""
        conn = self._conn()
        conn.request("GET", "/nonexistent")
        resp = conn.getresponse()
        self.assertEqual(resp.status, 404)
        conn.close()

    # ----- CORS preflight -----

    def test_options_cors_preflight(self):
        """OPTIONS request should return CORS headers."""
        conn = self._conn()
        conn.request("OPTIONS", "/v1/models")
        resp = conn.getresponse()
        self.assertIn(resp.status, (200, 204))
        allow_origin = resp.getheader("Access-Control-Allow-Origin")
        self.assertIsNotNone(allow_origin)
        conn.close()

    # ----- malformed request -----

    def test_malformed_body_returns_error(self):
        """POST with invalid JSON body should not crash the server."""
        conn = self._conn()
        conn.request(
            "POST",
            "/v1/chat/completions",
            body="THIS IS NOT JSON{{{{",
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        self.assertIn(resp.status, (400, 404, 500))
        conn.close()

    # ----- server still alive after all tests -----

    def test_zz_server_still_alive(self):
        """After all other tests, the server should still respond."""
        conn = self._conn()
        conn.request("GET", "/health")
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)
        conn.close()


if __name__ == "__main__":
    unittest.main()
