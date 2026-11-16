"""
Live view — an MJPEG page on localhost, an optional desktop window, and the
name prompt that appears when an unknown person is seen.

The page is the main surface: it works from any device on the network, so the
stream can be put on a projector or watched on a phone while standing in front
of the robot. The desktop window is there for when nothing else is to hand.

The name prompt is the "pop up". When the follow loop meets a face it does not
recognise, a field appears on the page; typing a name and submitting it enrols
that person. The same name can be typed into the terminal instead — both feed
one slot, and whichever arrives first wins.

Threading: the HTTP server runs in its own threads and only ever reads the
latest JPEG under a lock. The follow loop never blocks on a viewer, and a
viewer that disconnects mid-frame cannot stall the robot.

Standalone, with no robot (serves a test pattern):
    python live_view/live_view.py
    then open http://localhost:8080
"""

import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

DEFAULT_PORT = 8080
JPEG_QUALITY = 80
BOUNDARY = "k1frame"

# Colours are BGR, because that is what OpenCV draws in.
COLOUR_LOCKED = (90, 210, 90)
COLOUR_UNKNOWN = (60, 170, 250)
COLOUR_OTHER = (170, 170, 170)
COLOUR_TEXT = (245, 245, 245)

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>K1 Person Follow</title>
<style>
  :root { color-scheme: dark; --bg:#14161a; --panel:#1d2026; --line:#2c313a;
          --text:#e8eaed; --dim:#9aa2ad; --ok:#5ad15a; --ask:#fab040; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:15px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif; }
  .wrap { max-width: 1120px; margin:0 auto; padding:16px; }
  h1 { font-size:17px; font-weight:600; margin:0 0 12px; letter-spacing:.01em; }
  h1 span { color:var(--dim); font-weight:400; }
  .stream { width:100%; border-radius:10px; border:1px solid var(--line);
            background:#000; display:block; }
  .bar { display:flex; flex-wrap:wrap; gap:8px 20px; margin-top:12px;
         padding:12px 14px; background:var(--panel); border:1px solid var(--line);
         border-radius:10px; font-variant-numeric: tabular-nums; }
  .bar div { min-width:92px; }
  .bar b { display:block; color:var(--dim); font-size:11px; font-weight:500;
           text-transform:uppercase; letter-spacing:.06em; }
  #ask { display:none; margin-top:12px; padding:14px;
         background:var(--panel); border:1px solid var(--ask); border-radius:10px; }
  #ask.show { display:block; }
  #ask p { margin:0 0 10px; color:var(--ask); font-weight:600; }
  form { display:flex; gap:8px; flex-wrap:wrap; }
  input { flex:1 1 220px; padding:9px 11px; border-radius:7px;
          border:1px solid var(--line); background:#14161a; color:var(--text);
          font-size:15px; }
  button { padding:9px 18px; border:0; border-radius:7px; background:var(--ok);
           color:#0d1a0d; font-weight:600; font-size:15px; cursor:pointer; }
  button.ghost { background:var(--line); color:var(--text); }
  .note { margin-top:10px; color:var(--dim); font-size:13px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>K1 Person Follow <span id="mode"></span></h1>
  <img class="stream" src="/stream.mjpg" alt="live camera">

  <div class="bar">
    <div><b>State</b><span id="state">-</span></div>
    <div><b>Person</b><span id="name">-</span></div>
    <div><b>Track</b><span id="track">-</span></div>
    <div><b>Frame rate</b><span id="fps">-</span></div>
    <div><b>Frame age</b><span id="age">-</span></div>
    <div><b>Head</b><span id="head">-</span></div>
    <div><b>Arm</b><span id="arm">-</span></div>
    <div><b>Enrolled</b><span id="enrolled">-</span></div>
  </div>

  <div id="ask">
    <p>Unknown person in view — who is this?</p>
    <form onsubmit="return submitName(event)">
      <input id="who" placeholder="Type their name" autocomplete="off" required>
      <button type="submit">Save &amp; follow</button>
      <button type="button" class="ghost" onclick="skip()">Skip</button>
    </form>
    <div class="note">Saved to this machine only, for recognising them next run.</div>
  </div>
</div>

<script>
function put(id, value) { document.getElementById(id).textContent = value; }

async function poll() {
  try {
    const status = await (await fetch('/status', {cache: 'no-store'})).json();
    put('state', status.state || '-');
    put('name', status.name || '-');
    put('track', status.track_id === null ? '-' : status.track_id);
    put('fps', status.fps ? status.fps.toFixed(1) + ' Hz' : '-');
    put('age', status.age_ms === null ? '-' : Math.round(status.age_ms) + ' ms');
    put('head', status.head || '-');
    put('arm', status.arm || '-');
    put('enrolled', status.enrolled_count + ' saved');
    put('mode', status.dry_run ? '(dry run - head not moving)' : '');
    document.getElementById('ask').classList.toggle('show', !!status.asking);
    if (status.asking) document.getElementById('who').focus();
  } catch (error) { put('state', 'disconnected'); }
}

async function submitName(event) {
  event.preventDefault();
  const field = document.getElementById('who');
  const name = field.value.trim();
  if (!name) return false;
  await fetch('/name?value=' + encodeURIComponent(name), {method: 'POST'});
  field.value = '';
  return false;
}

async function skip() { await fetch('/name?skip=1', {method: 'POST'}); }

poll();
setInterval(poll, 500);
</script>
</body>
</html>
"""


def local_addresses(port):
    """URLs this page can be reached at, for printing at startup."""
    urls = ["http://localhost:%d" % port]
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))       # no packet is sent; this just picks a route
        urls.append("http://%s:%d" % (probe.getsockname()[0], port))
        probe.close()
    except OSError:
        pass
    return urls


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    view = None                      # set by LiveView before the server starts

    def log_message(self, *_args):
        pass                         # the robot's console is for the robot

    def _send(self, code, content_type, body, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", _PAGE.encode("utf-8"))
        elif path == "/status":
            body = json.dumps(self.view.status()).encode("utf-8")
            self._send(200, "application/json", body)
        elif path == "/stream.mjpg":
            self._stream()
        elif path == "/frame.jpg":
            jpeg = self.view.current_jpeg()
            if jpeg is None:
                self._send(503, "text/plain", b"no frame yet")
            else:
                self._send(200, "image/jpeg", jpeg)
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/name":
            self._send(404, "text/plain", b"not found")
            return
        query = parse_qs(parsed.query)
        if query.get("skip"):
            self.view.submit_name("")            # empty means "skip this person"
        else:
            value = (query.get("value") or [""])[0].strip()
            if value:
                self.view.submit_name(value)
        self._send(200, "application/json", b'{"ok":true}')

    def _stream(self):
        """A multipart MJPEG response: the oldest trick that just works everywhere."""
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=%s" % BOUNDARY)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while not self.view.stopped:
                jpeg = self.view.current_jpeg()
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--" + BOUNDARY.encode() + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(b"Content-Length: %d\r\n\r\n" % len(jpeg))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                time.sleep(1.0 / 30.0)
        except (BrokenPipeError, ConnectionResetError):
            return                   # the viewer closed the tab; nothing to report


class LiveView:
    """Publishes annotated frames and collects a typed name.

    publish() is called from the follow loop with the annotated frame. The HTTP
    threads only read the stored JPEG, so a slow or absent viewer never slows
    the robot down.
    """

    def __init__(self, port=DEFAULT_PORT, serve_web=True, window=False):
        self.port = port
        self.serve_web = serve_web
        self.window = window
        self.stopped = False

        self._lock = threading.Lock()
        self._jpeg = None
        self._status = {
            "state": "starting", "name": None, "track_id": None, "fps": 0.0,
            "age_ms": None, "head": "-", "arm": "off", "enrolled_count": 0,
            "asking": False, "dry_run": False,
        }
        self._pending_name = None
        self._name_event = threading.Event()

        self._server = None
        self._server_thread = None
        self._stdin_thread = None
        self.urls = []

        if serve_web:
            self._start_server()
        if window:
            self._start_stdin_reader()

    # -- web ------------------------------------------------------------------
    def _start_server(self):
        handler = type("_BoundHandler", (_Handler,), {"view": self})
        self._server = ThreadingHTTPServer(("0.0.0.0", self.port), handler)
        self._server.daemon_threads = True
        self._server_thread = threading.Thread(target=self._server.serve_forever,
                                               daemon=True, name="k1_live_view")
        self._server_thread.start()
        self.urls = local_addresses(self.port)

    # -- terminal fallback ----------------------------------------------------
    def _start_stdin_reader(self):
        """Read names typed into the terminal, for when the page is not open.

        A daemon thread blocked on input() cannot be interrupted, which is why
        it is a daemon: the process exits regardless of it.
        """
        def read():
            while not self.stopped:
                try:
                    line = sys.stdin.readline()
                except (OSError, ValueError):
                    return
                if not line:
                    return
                line = line.strip()
                if line and self._status.get("asking"):
                    self.submit_name(line)

        self._stdin_thread = threading.Thread(target=read, daemon=True,
                                              name="k1_name_stdin")
        self._stdin_thread.start()

    # -- frames ---------------------------------------------------------------
    def publish(self, bgr):
        """Store this frame as the one viewers see. Encodes once, not per viewer."""
        import cv2

        ok, encoded = cv2.imencode(".jpg", bgr,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return
        with self._lock:
            self._jpeg = encoded.tobytes()

    def current_jpeg(self):
        with self._lock:
            return self._jpeg

    # -- status ---------------------------------------------------------------
    def set_status(self, **fields):
        with self._lock:
            self._status.update(fields)

    def status(self):
        with self._lock:
            return dict(self._status)

    # -- the name prompt ------------------------------------------------------
    def ask_name(self, asking=True):
        """Show or hide the name field."""
        self.set_status(asking=bool(asking))

    def submit_name(self, name):
        """Called from an HTTP thread or the stdin thread. Empty means skip."""
        with self._lock:
            self._pending_name = name
        self._name_event.set()

    def take_name(self):
        """The submitted name if one arrived, else None. Clears the slot.

        Returns "" when the viewer pressed Skip, which the caller treats as
        "do not enrol this person" rather than "nothing happened".
        """
        if not self._name_event.is_set():
            return None
        with self._lock:
            name = self._pending_name
            self._pending_name = None
        self._name_event.clear()
        return name

    # -- desktop window -------------------------------------------------------
    def pump_window(self, bgr):
        """Draw the window and return the key pressed, or -1. No-op if disabled.

        Returns -1 when the window cannot be created, so a headless machine
        degrades to web-only instead of crashing.
        """
        if not self.window:
            return -1
        import cv2

        try:
            cv2.imshow("K1 Person Follow", bgr)
            return cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            print("live view: window unavailable (%s); continuing web-only" % exc)
            self.window = False
            return -1

    def close(self):
        self.stopped = True
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self.window:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Drawing the overlay
# -----------------------------------------------------------------------------
def annotate(bgr, people=None, locked_id=None, locked_name=None, face_box=None,
             aim=None, banner=None, bearing_rad=None):
    """The annotated copy of a frame that viewers see.

    Draws every detected person, highlights the locked one, marks the aim point,
    and shows the bearing the head and arm are using.
    """
    import cv2

    canvas = bgr.copy()
    height, width = canvas.shape[:2]

    for person in people or []:
        x1, y1, x2, y2 = person.box
        is_locked = locked_id is not None and person.track_id == locked_id
        if is_locked:
            colour, thickness = COLOUR_LOCKED, 3
            label = locked_name or "locked"
        elif locked_id is None:
            colour, thickness = COLOUR_UNKNOWN, 2
            label = "person %s" % (person.track_id if person.track_id is not None else "?")
        else:
            colour, thickness = COLOUR_OTHER, 1
            label = ""
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)
        if label:
            cv2.putText(canvas, label, (x1, max(16, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)

    if face_box is not None:
        fx1, fy1, fx2, fy2 = face_box
        cv2.rectangle(canvas, (fx1, fy1), (fx2, fy2), COLOUR_LOCKED, 1)

    # Frame centre: where the head is trying to put the person.
    cv2.drawMarker(canvas, (width // 2, height // 2), COLOUR_OTHER,
                   cv2.MARKER_CROSS, 18, 1)

    if aim is not None:
        ax, ay = int(aim[0]), int(aim[1])
        cv2.circle(canvas, (ax, ay), 7, COLOUR_LOCKED, 2)
        cv2.line(canvas, (width // 2, height // 2), (ax, ay), COLOUR_LOCKED, 1)

    if bearing_rad is not None:
        cv2.putText(canvas, "bearing %+.2f rad" % bearing_rad,
                    (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    COLOUR_TEXT, 1, cv2.LINE_AA)

    if banner:
        cv2.rectangle(canvas, (0, 0), (width, 34), (28, 30, 36), -1)
        cv2.putText(canvas, banner, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    COLOUR_TEXT, 1, cv2.LINE_AA)
    return canvas


if __name__ == "__main__":
    import numpy as np

    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    view = LiveView(port=port, serve_web=True, window=False)
    print("Test pattern. Open:")
    for url in view.urls:
        print("  %s" % url)
    print("The name field appears after 5 seconds. Ctrl+C to stop.")

    try:
        start = time.monotonic()
        tick = 0
        while True:
            tick += 1
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            frame[:] = (24, 26, 30)
            shift = int((time.monotonic() * 120) % 640)
            frame[:, max(0, shift - 40):shift] = (90, 210, 90)
            elapsed = time.monotonic() - start
            view.publish(frame)
            view.set_status(state="test pattern", fps=tick / max(elapsed, 0.01),
                            age_ms=0, head="0.00 / 0.00", enrolled_count=0)
            if elapsed > 5:
                view.ask_name(True)
            name = view.take_name()
            if name is not None:
                print("got name from page: %r" % name)
                view.ask_name(False)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        view.close()
