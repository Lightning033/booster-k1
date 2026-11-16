"""
A live window into the detector while it runs — served on this machine.

What it shows, updating as the robot watches somebody:

  * the person-class response of each of YOLO's three detection scales, drawn
    as heatmaps, so you can watch which branch is doing the work
  * the peak confidence per scale and which one is winning
  * the recognition side: who is locked, and the cosine score behind it
  * frame rate and frame age

How it gets the data without slowing the loop down. YOLO already computes the
class scores for every frame; they are simply thrown away after the boxes are
decoded. A forward hook on the head's class branch keeps a reference to those
tensors as they go past. Nothing extra is computed, and the measured cost is
inside the noise of a 70 ms inference.

One trap, found by testing rather than by reading: ultralytics copies the model
into its predictor on the first inference, and a module's hooks are copied with
it. Hooks registered before the first run therefore land on a model that is no
longer the one executing, and removing them does nothing. So ScaleTap attaches
LAZILY, to m.predictor.model, after the first inference has happened.

The page is read-only. It cannot command the robot.

Standalone, with a synthetic pattern and no robot:
    python net_view/net_view.py
"""

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 8091
SCALES = (("P3", 8), ("P4", 16), ("P5", 32))

# The page polls about three times a second, so encoding a new state on every
# frame is work nobody reads. Measured: encoding costs ~8 ms, which is 10% of a
# 12 Hz frame budget in a loop the head's timing depends on. Throttling to the
# page's own rate drops the average cost to about 2 ms per frame.
MIN_PUBLISH_INTERVAL_S = 0.28

# Nodes drawn per stage in the graph. Each is a bucket of channels: a node per
# neuron is not a drawing anybody can read.
NODES_PER_STAGE = 12


class LayerTap:
    """Per-stage activations from every stage of the detector, for the graph.

    Each node drawn on the page is a BUCKET of channels, not one neuron: the
    network holds millions of activations and 3.16M weights, so a node per
    neuron is not a drawing anybody can read. A bucket's value is the mean
    absolute activation of the channels in it.

    The dense lines between columns are real. Every convolution in this model
    has groups=1, which means each output channel is computed from ALL input
    channels, so a line from every node of one stage to every node of the next
    reflects a dependency that genuinely exists. Were any conv depthwise, that
    would stop being true and the lines would have to go.

    Attaches lazily for the same reason as ScaleTap: the model that runs is the
    predictor's copy, which does not exist until the first inference.
    """

    def __init__(self, detector, nodes=NODES_PER_STAGE):
        self.detector = detector
        self.nodes = int(nodes)
        self.hooks = []
        self.attached = False
        self.failed = False
        self.edges = []
        self._acts = {}
        self._lock = threading.Lock()

    def _stages(self):
        model = getattr(self.detector, "model", None)
        predictor = getattr(model, "predictor", None)
        if predictor is None:
            return None
        backend = getattr(predictor, "model", None)
        inner = getattr(backend, "model", backend)
        return getattr(inner, "model", None)

    def maybe_attach(self):
        if self.attached or self.failed:
            return self.attached
        stages = self._stages()
        if stages is None:
            return False
        try:
            import torch                                  # noqa: F401
        except ImportError:
            self.failed = True
            return False

        def make(index):
            def capture(module, inputs, output):
                self._store(index, output)
            return capture

        try:
            for index, module in enumerate(stages):
                self.hooks.append(module.register_forward_hook(make(index)))
            self.edges = self._read_edges(stages)
        except Exception:
            self.detach()
            self.failed = True
            return False
        self.attached = True
        return True

    @staticmethod
    def _read_edges(stages):
        """The real wiring: which stage feeds which, from the model itself."""
        edges = []
        for index, module in enumerate(stages):
            sources = getattr(module, "f", -1)
            if not isinstance(sources, (list, tuple)):
                sources = [sources]
            for source in sources:
                origin = index - 1 if source == -1 else int(source)
                if 0 <= origin < len(stages):
                    edges.append([origin, index, 1 if source == -1 else 0])
        return edges

    def _store(self, index, output):
        import torch

        if not isinstance(output, torch.Tensor) or output.dim() != 4:
            return
        try:
            with torch.no_grad():
                per_channel = output[0].abs().mean(dim=(1, 2))
                count = per_channel.numel()
                if count >= self.nodes:
                    usable = (count // self.nodes) * self.nodes
                    buckets = per_channel[:usable].view(self.nodes, -1).mean(dim=1)
                else:
                    buckets = per_channel
                values = buckets.cpu().numpy()
        except Exception:
            return
        with self._lock:
            self._acts[index] = values

    def latest(self):
        """{"stages": [{i, name, v: [0..1]}], "edges": [[from, to, direct]]}"""
        with self._lock:
            if not self._acts:
                return None
            acts = {index: values.copy() for index, values in self._acts.items()}
        stages = []
        for index in sorted(acts):
            values = acts[index]
            top = float(values.max()) or 1.0
            stages.append({"i": index,
                           "v": [round(float(value) / top, 3) for value in values]})
        return {"stages": stages, "edges": list(self.edges)}

    def detach(self):
        for hook in self.hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self.hooks = []
        self.attached = False
        with self._lock:
            self._acts.clear()


class ScaleTap:
    """Keeps the latest per-scale person-class response from a PersonDetector.

    Attaches itself the first time maybe_attach() is called after the detector
    has run at least once. Until then latest() returns None and the caller
    carries on; a tap that cannot attach is never fatal.
    """

    def __init__(self, detector, class_index=0):
        self.detector = detector
        self.class_index = class_index
        self.hooks = []
        self.attached = False
        self.failed = False
        self._maps = {}
        self._lock = threading.Lock()

    # -- attaching ------------------------------------------------------------
    def _detect_module(self):
        """The Detect module of the model that actually runs, or None."""
        model = getattr(self.detector, "model", None)
        predictor = getattr(model, "predictor", None)
        if predictor is None:
            return None                      # no inference has happened yet
        backend = getattr(predictor, "model", None)
        inner = getattr(backend, "model", backend)
        sequential = getattr(inner, "model", None)
        if sequential is None:
            return None
        try:
            head = sequential[-1]
        except (TypeError, IndexError):
            return None
        return head if hasattr(head, "cv3") else None

    def maybe_attach(self):
        """Try to attach. True once attached; False while not possible yet."""
        if self.attached or self.failed:
            return self.attached
        head = self._detect_module()
        if head is None:
            return False
        try:
            import torch                                      # noqa: F401
        except ImportError:
            self.failed = True
            return False

        def make(index):
            def capture(module, inputs, output):
                self._store(index, output)
            return capture

        try:
            for index in range(len(head.cv3)):
                self.hooks.append(head.cv3[index].register_forward_hook(make(index)))
        except Exception:
            self.detach()
            self.failed = True
            return False
        self.attached = True
        return True

    def _store(self, index, output):
        import torch

        try:
            with torch.no_grad():
                plane = torch.sigmoid(output[0, self.class_index]).cpu().numpy()
        except Exception:
            return
        with self._lock:
            self._maps[index] = plane

    def latest(self):
        """[(name, stride, 2-D array)] newest first captured, or None."""
        with self._lock:
            if not self._maps:
                return None
            items = []
            for index in sorted(self._maps):
                name, stride = (SCALES[index] if index < len(SCALES)
                                else ("P%d" % (index + 3), 8 << index))
                items.append((name, stride, self._maps[index]))
            return items

    def detach(self):
        for hook in self.hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self.hooks = []
        self.attached = False
        with self._lock:
            self._maps.clear()


def encode_grid(grid):
    """A 2-D float array of 0..1 as {w, h, d} with d base64 uint8."""
    import numpy as np

    array = np.asarray(grid, dtype="float32")
    if array.ndim != 2 or array.size == 0:
        return None
    clipped = np.clip(array, 0.0, 1.0)
    as_bytes = (clipped * 255.0 + 0.5).astype("uint8")
    return {"w": int(array.shape[1]), "h": int(array.shape[0]),
            "peak": round(float(array.max()), 4),
            "d": base64.b64encode(as_bytes.tobytes()).decode("ascii")}


class NetView:
    """Serves the page and the state. Failing to bind is never fatal."""

    def __init__(self, port=DEFAULT_PORT, serve=True, stream_url=None,
                 attempts=8, min_interval_s=MIN_PUBLISH_INTERVAL_S):
        self.port = int(port)
        self.serve = bool(serve)
        self.stream_url = stream_url
        self.min_interval_s = float(min_interval_s)
        self._state = {"scales": [], "stats": {}}
        self._lock = threading.Lock()
        self._server = None
        self._thread = None
        self._last_publish = 0.0
        self.skipped = 0
        self.published = 0
        self.urls = []
        if self.serve:
            self._start(attempts)

    # -- serving --------------------------------------------------------------
    def _start(self, attempts):
        view = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass                                   # keep the console clean

            def _send(self, body, content_type):
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
                elif path == "/state.json":
                    self._send(json.dumps(view.snapshot()).encode("utf-8"),
                               "application/json")
                else:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

        last_error = None
        for offset in range(max(1, attempts)):
            try:
                self._server = ThreadingHTTPServer(("0.0.0.0", self.port + offset),
                                                   Handler)
                self.port = self.port + offset
                break
            except OSError as exc:
                last_error = exc
                self._server = None
        if self._server is None:
            print("  net view: could not bind a port (%s); carrying on without it"
                  % last_error)
            self.serve = False
            return
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True, name="k1_net_view")
        self._thread.start()
        self.urls = ["http://localhost:%d" % self.port]

    # -- state ----------------------------------------------------------------
    def publish(self, tap_items, stats=None, force=False, layers=None):
        """Store the newest maps and numbers.

        Called from the control loop every tick, so it drops any call that
        arrives sooner than the page can use it. Returns True when the state
        was actually rebuilt.
        """
        now = time.monotonic()
        if not force and now - self._last_publish < self.min_interval_s:
            self.skipped += 1
            if stats:                       # numbers are cheap; keep them fresh
                with self._lock:
                    self._state.setdefault("stats", {}).update(stats)
            return False
        self._last_publish = now
        self.published += 1
        scales = []
        if tap_items:
            for name, stride, grid in tap_items:
                encoded = encode_grid(grid)
                if encoded is None:
                    continue
                encoded["name"] = name
                encoded["stride"] = stride
                scales.append(encoded)
        with self._lock:
            self._state = {"scales": scales, "stats": dict(stats or {}),
                           "stream": self.stream_url, "t": time.time(),
                           "graph": layers}
        return True

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def close(self):
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>K1 detector — live</title>
<style>
 :root{--bg:#0C1417;--panel:#141F24;--line:#243439;--fg:#E7EEEC;--muted:#93A6A6;
       --accent:#3FB8A0;--hot:#F07B5F;
       --mono:'IBM Plex Mono',ui-monospace,Menlo,Consolas,monospace}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
 .wrap{max-width:1180px;margin:0 auto;padding:22px max(16px,env(safe-area-inset-left,0px)) 40px}
 h1{font-size:19px;margin:0 0 3px;letter-spacing:-.01em}
 .sub{color:var(--muted);font-size:13px;margin:0 0 20px}
 .row{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px}
 .stat{background:var(--panel);border:1px solid var(--line);border-radius:9px;
       padding:11px 14px;min-width:116px;flex:1 1 116px}
 .stat b{display:block;font-family:var(--mono);font-size:21px;font-variant-numeric:tabular-nums}
 .stat span{display:block;color:var(--muted);font-size:11.5px;letter-spacing:.06em;
            text-transform:uppercase;margin-top:3px}
 .grids{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px;min-width:0}
 .card.win{border-color:var(--hot)}
 .card h2{font-family:var(--mono);font-size:13px;margin:0 0 2px;font-weight:600}
 .card .meta{font-family:var(--mono);font-size:11.5px;color:var(--muted);margin-bottom:9px}
 .card.win .meta{color:var(--hot)}
 canvas{width:100%;height:auto;display:block;image-rendering:pixelated;
        border-radius:6px;background:#000}
 .graphbox{background:var(--panel);border:1px solid var(--line);border-radius:10px;
            padding:13px 13px 10px;margin-bottom:16px}
 .gh{display:flex;justify-content:space-between;align-items:baseline;gap:12px;
     flex-wrap:wrap;margin-bottom:9px;font-family:var(--mono);font-size:13px;font-weight:600}
 .gout{color:var(--hot);font-family:var(--mono);font-size:15px}
 .gscroll{overflow-x:auto}
 #graph{display:block;height:330px;min-width:980px;background:#0a1114;border-radius:7px}
 .gkey{font-size:12px;color:var(--muted);margin-top:8px;line-height:1.45}
 .live{margin-bottom:16px}
 .live img{width:100%;max-width:560px;border:1px solid var(--line);border-radius:10px;display:block}
 .foot{color:var(--muted);font-size:12.5px;margin-top:22px;max-width:70ch}
 .off{color:var(--hot)}
</style>
<div class="wrap">
<h1>Detector, live</h1>
<p class="sub">Person-class response of each detection scale, read from the forward pass as it happens.</p>
<div class="row" id="stats"></div>
<div class="graphbox">
  <div class="gh"><span id="g-title">Every stage, lit by its own activation</span>
    <span id="g-out" class="gout"></span></div>
  <div class="gscroll"><canvas id="graph"></canvas></div>
  <div class="gkey" id="g-key"></div>
</div>
<div class="live" id="live"></div>
<div class="grids" id="grids"></div>
<p class="foot" id="foot"></p>
</div>
<script>
const COLOR=[];
for(let i=0;i<256;i++){           // inferno-ish ramp
  const t=i/255;
  COLOR.push([Math.round(255*Math.min(1,Math.max(0,1.35*t-0.06))),
              Math.round(255*Math.min(1,Math.max(0,1.5*t*t-0.1))),
              Math.round(255*Math.min(1,Math.max(0,0.45*Math.sin(3.14159*t)+1.6*Math.pow(t,6))))]);
}
function draw(cv, s, scale){
  cv.width=s.w; cv.height=s.h;
  const ctx=cv.getContext("2d"), img=ctx.createImageData(s.w,s.h);
  const bin=atob(s.d);
  for(let i=0;i<s.w*s.h;i++){
    const v=Math.min(255, Math.round(bin.charCodeAt(i)*scale));
    const c=COLOR[v];
    img.data[i*4]=c[0]; img.data[i*4+1]=c[1]; img.data[i*4+2]=c[2]; img.data[i*4+3]=255;
  }
  ctx.putImageData(img,0,0);
}
const gcv=document.getElementById("graph");
const BAND=i=>i<=9?"backbone":(i<=21?"neck":"head");
const BANDC={backbone:[63,184,160],neck:[240,123,95],head:[111,180,212]};
function drawGraph(g){
  const dpr=window.devicePixelRatio||1;
  const w=Math.max(980, gcv.clientWidth), h=330;
  gcv.width=w*dpr; gcv.height=h*dpr;
  const c=gcv.getContext("2d"); c.setTransform(dpr,0,0,dpr,0,0);
  c.clearRect(0,0,w,h);
  if(!g||!g.stages||!g.stages.length) return;
  const n=g.stages.length, padL=26, padR=96;
  const colW=(w-padL-padR)/Math.max(1,n-1);
  const pos={};
  g.stages.forEach((st,k)=>{
    const x=padL+k*colW, m=st.v.length;
    const top=34, bot=h-46, span=bot-top;
    pos[st.i]=st.v.map((v,j)=>({x:x, y:top+(m===1?span/2:j*span/(m-1)), v:v}));
  });
  // edges first, so nodes paint over them
  (g.edges||[]).forEach(([from,to,direct])=>{
    const a=pos[from], b=pos[to];
    if(!a||!b) return;
    const col=BANDC[BAND(to)];
    if(direct){
      // every output channel depends on every input channel: draw the mesh
      for(const p of a) for(const q of b){
        const strength=p.v*q.v;
        if(strength<0.035) continue;
        c.strokeStyle="rgba("+col[0]+","+col[1]+","+col[2]+","+(strength*0.4).toFixed(3)+")";
        c.lineWidth=0.6+strength*1.1;
        c.beginPath(); c.moveTo(p.x,p.y); c.lineTo(q.x,q.y); c.stroke();
      }
    }else{
      // a skip connection: one curve, so it reads as the long jump it is
      const p=a[Math.floor(a.length/2)], q=b[Math.floor(b.length/2)];
      c.strokeStyle="rgba(230,230,235,0.5)"; c.lineWidth=1.6;
      c.setLineDash([5,4]);
      c.beginPath(); c.moveTo(p.x,p.y);
      c.quadraticCurveTo((p.x+q.x)/2,Math.min(p.y,q.y)-38,q.x,q.y); c.stroke();
      c.setLineDash([]);
    }
  });
  g.stages.forEach((st)=>{
    const col=BANDC[BAND(st.i)];
    pos[st.i].forEach(p=>{
      c.beginPath(); c.arc(p.x,p.y,5.4,0,6.2832);
      c.fillStyle="rgba("+col[0]+","+col[1]+","+col[2]+","+(0.10+p.v*0.9).toFixed(3)+")";
      c.fill();
      if(p.v>0.55){ c.strokeStyle="rgba(255,255,255,"+((p.v-0.55)*1.4).toFixed(2)+")";
                    c.lineWidth=1; c.stroke(); }
    });
    c.fillStyle="#6b8086"; c.font="10px ui-monospace,monospace"; c.textAlign="center";
    c.fillText(st.i, pos[st.i][0].x, h-30);
  });
  c.textAlign="left"; c.fillStyle="#93A6A6"; c.font="11px ui-monospace,monospace";
  c.fillText("backbone 0-9", padL, h-12);
  c.fillText("neck 10-21", padL+colW*10, h-12);
  c.fillText("head 22", padL+colW*21.4, h-12);
}
let haveStream=false;
async function tick(){
  let st;
  try{ st=await (await fetch("state.json",{cache:"no-store"})).json(); }
  catch(e){ document.getElementById("foot").innerHTML='<span class="off">Lost contact with the run.</span>'; return; }
  const s=st.stats||{};
  document.getElementById("stats").innerHTML=[
    ["fps", (s.fps!=null? s.fps.toFixed(1):"—"), "frames / s"],
    ["age", (s.age_ms!=null? Math.round(s.age_ms)+" ms":"—"), "frame age"],
    ["people", (s.people!=null? s.people:"—"), "detected"],
    ["who", (s.name||"—"), "locked"],
    ["score", (s.score!=null? s.score.toFixed(2):"—"), "face match"]
  ].map(([k,v,l])=>'<div class="stat"><b>'+v+'</b><span>'+l+'</span></div>').join("");

  if(st.stream && !haveStream){
    document.getElementById("live").innerHTML='<img alt="live camera view" src="'+st.stream+'">';
    haveStream=true;
  }
  drawGraph(st.graph);
  const scales=st.scales||[];
  const peak=Math.max(0.0001,...scales.map(x=>x.peak));
  const host=document.getElementById("grids");
  if(host.children.length!==scales.length){
    host.innerHTML=scales.map(x=>'<div class="card"><h2>'+x.name+'</h2>'+
      '<div class="meta"></div><canvas></canvas></div>').join("");
  }
  scales.forEach((x,i)=>{
    const card=host.children[i];
    card.classList.toggle("win", x.peak>=peak-1e-9 && x.peak>0.2);
    card.querySelector("h2").textContent=x.name+" · stride "+x.stride;
    card.querySelector(".meta").textContent=x.w+"×"+x.h+" grid · peak "+x.peak.toFixed(3);
    draw(card.querySelector("canvas"), x, 255/ (peak*255));
  });
  const best=scales.length?Math.max(...scales.map(x=>x.peak)):0;
  document.getElementById("g-out").textContent =
    best>0.5 ? '-> "person"  '+best.toFixed(2) : (best>0?'-> nothing confident  '+best.toFixed(2):"");
  document.getElementById("g-key").textContent =
    "Each node is a group of channels, lit by its own mean activation — not one neuron: "
    + "the model holds 3.16M weights. Every convolution here has groups=1, so each output "
    + "channel really is computed from every input channel, which is what the mesh between "
    + "columns shows. Dashed curves are the neck's skip connections.";
  document.getElementById("foot").textContent= scales.length
    ? "All panels share one colour scale, so a dark panel means that branch is genuinely quiet rather than merely dimmer."
    : "Waiting for the first inference — the tap attaches after the detector has run once.";
}
tick(); setInterval(tick, 320);
</script>
"""


if __name__ == "__main__":
    import math
    import numpy as np

    view = NetView()
    if not view.serve:
        raise SystemExit("could not serve")
    print("Synthetic pattern, no robot. Open:")
    for url in view.urls:
        print("  %s" % url)
    print("Ctrl+C to stop.")
    try:
        step = 0
        while True:
            step += 1
            items = []
            for index, (name, stride) in enumerate(SCALES):
                height, width = 48 >> index, 80 >> index
                yy, xx = np.mgrid[0:height, 0:width]
                centre = (math.sin(step / 12.0) * 0.3 + 0.5) * width
                blob = np.exp(-(((xx - centre) ** 2) / (width * 0.6) +
                                ((yy - height / 2.0) ** 2) / (height * 0.6)))
                weight = 0.9 if index == 2 else 0.04
                items.append((name, stride, (blob * weight).astype("float32")))
            view.publish(items, {"fps": 12.0, "age_ms": 14, "people": 1,
                                 "name": "demo", "score": 0.87})
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        view.close()
