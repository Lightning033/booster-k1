# k1-person-follow

**Overview:** Vision subsystem for the Booster K1 EDU humanoid. It picks one
person out of a room, learns their name the first time it sees them,
recognises them on every later run, and keeps the robot's head pointed at
them. The data flow is: robot camera → camera process (ROS2) → JPEG over a
pipe → pipeline process (Booster SDK) → head commands.

**The robot does not walk.** No locomotion command is sent at any point, not
even a zero one. A test asserts this.

---

## Key design decisions

- **ROS2 and the Booster SDK never share a process.** Sharing one loses camera
  frames; on the previous project depth frames stopped arriving entirely while
  colour kept working. The camera process imports `rclpy` and never the SDK;
  the pipeline process imports the SDK and never `rclpy`. The split also
  enforces the SDK's one-initialisation-per-process rule, which a second call
  turns into a segfault.
- **The lock is an identity, not a tracker id.** ByteTrack issues a new track
  id after a single dropped detection, which froze the head for five seconds
  with the person in plain view. The face is now the thing being followed and
  the track id is a per-frame shortcut.
- **No depth in the loop.** The camera rides on the head, so image error
  converts straight into a head angle through the focal length. No depth, no
  world frame, no calibration rig.
- **Lock first, ask second.** The head starts tracking on the first frame;
  identification happens afterwards, so nothing waits on a person typing.
- **The camera topic is discovered, not assumed.** On this robot the
  compressed topic advertises at 30 Hz and delivers nothing to a subscriber,
  so the node probes, finds no frames and falls back to raw by itself.
- **The arm is opt-in.** Its torso-frame axes are not verified on this robot
  and a sign error sends the limb the wrong way.

## Pipeline

```
camera process (ROS2 only)  ──pipe (JPEG)──>  pipeline process (Booster SDK)
  topic + QoS discovery                        detect_person     YOLOv8n + ByteTrack
                                               recognize_face    YuNet + SFace
                                               face_db           names and IDs on disk
                                               follow_controller control law
                                               head_control      RotateHead
                                               arm_control       MoveHandEndEffectorV2
                                               live_view         MJPEG page
                                               net_view          detector graph
```

Per frame: read frame → find people → read the face → measure the error in
pixels → send one capped head step. A repeated frame is skipped; a frame older
than 600 ms produces no command at all.

---

## Setup and run

### What you need

**Hardware.** The K1 powered on with its ROS2 camera stack running, the lab
workstation on the robot's hotspot network, and a stand or a spotter — a mode
change can make the robot shift its feet.

**Network.** The robot answers at `192.168.5.3`. Its interface is `wlP5p1s0`,
the workstation's is `wlp5s0`. There is no internet on this network, so
everything must be on disk before starting — no model downloads at run time.

**Software.**

| Needs | Where it comes from |
| --- | --- |
| ROS2 Humble | `/opt/ros/humble` |
| Python 3.10 | system |
| `booster_robotics_sdk_python` | installed in `k1_env` |
| ultralytics, OpenCV, numpy < 2 | installed in `k1_follow_env` |
| YOLOv8n weights | `detect_person/yolov8n.pt` |
| YuNet and SFace ONNX models | `recognize_face/models/` |

**Why two environments.** `k1_env` holds the robot SDK. `k1_follow_env` holds
the vision stack and pins numpy below 2, because ROS2 and OpenCV 4.11 break
against numpy 2. Rather than merge them and risk one breaking the other, the
SDK is symlinked from the first into the second by `tools/link_sdk.py`.

### One-time setup

```bash
# 1. the vision environment — --system-site-packages is required,
#    or it cannot see ROS2 and the camera process will not run
python3 -m venv --system-site-packages ~/k1_follow_env
source ~/k1_follow_env/bin/activate
pip install "numpy<2" ultralytics opencv-python

# 2. link the SDK in (once per environment)
python tools/link_sdk.py          # prints VERIFIED on success

# 3. confirm the models are present
ls detect_person/yolov8n.pt
ls recognize_face/models/*.onnx
```

`link_sdk.py` finds `booster_robotics_sdk_python` wherever it is installed,
checks it was built for the same Python version, symlinks that one module into
the active environment and verifies the import in a fresh interpreter. Nothing
is copied or moved; `--unlink` reverses it. If the SDK was built for a
different Python version it refuses to link — a compiled extension cannot
cross versions.

Three model files are needed: `yolov8n.pt`,
`face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx`.

### Every session

```bash
cd <project root>
source /opt/ros/humble/setup.bash
source ~/k1_follow_env/bin/activate
export K1_NET_INTERFACE=wlp5s0
```

Source ROS2 **before** activating the environment. The other order leaves
ROS2's Python ahead of the environment's on the path and the wrong numpy is
imported.

Run everything from the project root — the modules import each other by
relative path.

### Checks, in order

Each is a checkpoint; if one fails there is no point running the next.

```bash
# 1. the SDK is reachable — no robot needed
python tools/link_sdk.py

# 2. the offline tests — no robot needed
cd tests && python -m unittest discover -s . 2>&1 | grep -E "^Ran |^OK$|^FAILED"; cd ..

# 3. the camera — robot on, walk mode not needed
ping -c 2 192.168.5.3
python camera_process/camera_process.py

# 4. walk mode — spotter present, space clear
python sdk_client/sdk_client.py          # reports the mode
python sdk_client/sdk_client.py walk     # switches to WALK

# 5. head directions — only after changing head_control.py; stand behind
python head_control/head_control.py yawtest     # +yaw turns LEFT
python head_control/head_control.py pitchtest   # +pitch tilts DOWN
```

Step 2 expects `Ran 269 tests` and `OK`. Filter for the summary rather than
using `tail`: the tests print to stdout and the summary to stderr.

In step 3, watch the **frame age**, not the frame rate. Under 300 ms is
healthy; an age that climbs means frames have stopped arriving.

In step 4 it must **name** a mode such as `kWalking`. If it prints an object
instead, the mode could not be read and the follow loop will refuse to start.
Head commands are rejected with error 400 in any other mode.

### Running it

```bash
# rehearsal — nothing moves, robot can be off
python pipeline/person_follow.py --lock-nearest --dry-run --window --port 8090

# the normal run
python pipeline/person_follow.py --lock-nearest --window --port 8090

# with the detector graph on a second port
python pipeline/person_follow.py --lock-nearest --net-view --window --port 8090
```

Open `localhost:8090` for the camera view and `localhost:8091` for the
detector graph. `Ctrl+C` exits cleanly — the head recentres, the arm lowers,
the camera closes. In the desktop window: `q` quit, `c` recentre, `u` unlock.

Dry run touches no SDK at all, and enrolment still works, so a person can be
registered before the robot is switched on. It is also the fallback if the
robot is uncooperative on the day.

| Flag | What it does |
| --- | --- |
| `--lock-nearest` | Lock the nearest person immediately; no enrolment first |
| `--dry-run` | Touch no SDK; enrolment still works |
| `--window` | Desktop window as well as the web page |
| `--port N` | Web page port (default 8080) |
| `--net-view` | Serve the detector graph on a second port |
| `--net-port N` | That port (default 8091) |
| `--speed N` | Head speed multiplier; default 0.4, 0.3 is gentler |
| `--no-hold` | Allow re-locking onto someone else; default holds one person |
| `--no-ask` | Never prompt for a name; known faces still recognised |
| `--raw` | Force the uncompressed camera topic |
| `--arm` | Enable hand pointing — **only after `frame_probe.py`** |

Before ever using `--arm`, run the read-only probe. It commands nothing:

```bash
python arm_control/frame_probe.py
```

---

## Enrolled people

The robot locks first and asks second. Once the face is wide enough to store,
a prompt appears in the terminal and on the page; typing a name saves the face
with an ID (`P001`, `P002`, …) assigned once and never reused.

Enrolment needs the face about **54 px wide, so inside roughly 1.9 m**.
Further away it still follows the person — it just will not save a sample it
cannot trust, because that sample is what every later run compares against.

```
face_db/people/<name>/
  meta.json          name, ID, sample count
  embeddings.npy     the face fingerprints, up to 12
  sample_01.jpg      the frame each sample came from
```

```bash
python face_db/face_db.py list
python face_db/face_db.py whois photo.jpg
python face_db/face_db.py remove harsh
python face_db/calibrate.py --pairs       # re-measure the threshold
```

**The threshold.** A match needs a cosine score of at least **0.55** with a
**0.05** margin over the runner-up. 0.40 was tried first and proved too loose:
a visitor scored 0.41 against an enrolled colleague and was briefly named as
them. Measured same-person scores stay above 0.70 even blurred, tilted or
under-lit, so 0.55 keeps a wide cushion. `calibrate.py` re-measures both sides
on whichever faces are enrolled; override with `K1_FACE_THRESHOLD`.

**One face, one name.** Saving the same person under two names makes every
later match ambiguous and silently ends recognition for them. The code detects
this and adds the new view to the existing person instead.

### Privacy

`face_db/people/` holds photographs of real people and the biometric data
derived from them.

- It is in `.gitignore`. Keep it out of any repository, shared drive or backup.
- Ask each person before enrolling them and say what it is for.
- Remove them afterwards with `face_db.py remove <slug>`.

---

## Testing

269 tests, all runnable with no robot attached:

```bash
cd tests && python -m unittest discover -s .
```

They cover the control law's sign conventions and convergence, the stale-frame
freeze, the database's threshold and margin gates, the camera pipe protocol
with real spawned processes, the ROS conversions including row padding and the
RGB/BGR swap, the identity re-acquisition path, and a bait trap proving no
locomotion command is ever sent.

**Measured, with the real models on real frames**

| Figure | Value |
| --- | --- |
| Frame rate in the loop | ~12 Hz (previous project: 4.9 Hz) |
| Frame age | under 30 ms |
| Worst same-person score vs best impostor | 0.79 vs 0.02 |
| Recognition range at the default detect width | 3.6 m |
| Time to centre a person 30° off axis | 1.5 s |

**Bugs this testing found before the robot did.** A dropped detection frame
renumbering the person and freezing the head; the robot's mode being read from
the wrong field, so the loop refused to start while the robot was correctly in
walk mode; one face saved under two names silently ending recognition; and a
`value or default` fallback that made `--speed 0` run at full speed. None of
the four raises an exception — each produces plausible behaviour that is
wrong, which is what a live demo cannot distinguish from working.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: booster_robotics_sdk_python` | Wrong environment, or never linked | Activate `k1_follow_env`, then `python tools/link_sdk.py` |
| `ModuleNotFoundError: rclpy` | Environment built without `--system-site-packages` | Rebuild it with that flag |
| numpy errors that never mention numpy | numpy 2 against ROS2 or OpenCV 4.11 | `pip install "numpy<2"` |
| `Address already in use` | A previous run still holds the port | `--port 8091`, or `pkill -f person_follow.py` |
| Camera reports no frames | The node lists every topic it tried | Read that report; it falls back on its own |
| Frame age climbing steadily | Frames have stopped arriving | Restart the camera process; check the WiFi link |
| Stuck on `searching` | Console names the reason, usually a face too small | Step closer; it prints the distance |
| Head does not move | `--dry-run` set, or not in walk mode | Drop the flag; re-run `sdk_client.py walk` |
| Error 400 on a head command | Robot left walk mode | `python sdk_client/sdk_client.py walk` |
| Head motion looks abrupt | Default speed too quick for the room | `--speed 0.3` |
| Wrong person recognised | Threshold too loose for these faces | Enrol everyone present, then `calibrate.py --pairs` |
| Robot unreachable | Hotspot dropped, or the K1 is asleep | `--dry-run`; the whole vision half still demonstrates |

None of these needs a code change — each is a flag, a reconnect, or a step
forward.

---

## Settings

Environment variables, so nothing needs editing to move machines. Only
`K1_NET_INTERFACE` normally has to be set.

| Variable | Default | What it is |
| --- | --- | --- |
| `K1_NET_INTERFACE` | `wlp5s0` | Interface that reaches the robot |
| `K1_DDS_DOMAIN` | `0` | DDS domain id |
| `K1_FACE_THRESHOLD` | `0.55` | Cosine score a match must reach |
| `K1_FACE_MARGIN` | `0.05` | Margin over the runner-up |
| `K1_FACE_DB` | `face_db/people` | Where enrolled people are stored |
| `K1_FACE_MODELS` | `recognize_face/models` | Where the ONNX files are |
| `K1_FACE_DETECT_WIDTH` | `960` | Face-detection width; higher sees further |
| `K1_YOLO_MODEL` | `detect_person/yolov8n.pt` | YOLO weights |

Detection width sets how far away a person can be recognised. Measured on a
1280×720 frame with fx = 644:

| Width | Smallest face | Usable range | Cost per frame |
| --- | --- | --- | --- |
| 640 | 39 px | 2.6 m | 10 ms |
| 960 (default) | 29 px | 3.6 m | 21 ms |
| 1280 | 23 px | 4.5 m | 36 ms |

---

## Scope and limitations

Head motion only — no locomotion, and the arm is disabled by default.

**Not verified**

- `PITCH_LIMIT = 0.35 rad` is deliberately conservative. The published K1 range
  is wider; raise it only after measuring.
- The arm's torso-frame axes. `frame_probe.py` exists to settle them, read-only.
  Everything around the arm is tested: the geometry, the hand choice, the rate
  limit, resting on shutdown, and that a refused pose switches the arm off and
  leaves the head following.
- Why the compressed camera topic advertises but delivers nothing to a
  subscriber on this robot. Not blocking — the node falls back to raw.
- Behaviour with more than three people at once; the search checks the three
  nearest.
- Recognition in rooms other than the one it was calibrated in.

**Known not working**

The robot cannot speak. It has no audio hardware connected — no driver bound at
`8-001c`, no USB-C media device and no iFlytek microphone array — and its TTS is
a cloud service with no internet route. Nothing in this module depends on sound;
names are shown on the page instead.

---

## Team

ICT Capstone project, University of Adelaide (PG-S2-43), with the AIML Robotics
Lab.

---

## Model files

Three model files are needed and are **not** in this repository — they are
third-party weights, and together they are about 45 MB.

| File | Where it goes | Source |
| --- | --- | --- |
| `yolov8n.pt` | `detect_person/` | ultralytics; downloaded on first use, or copied from a machine that has it |
| `face_detection_yunet_2023mar.onnx` | `recognize_face/models/` | OpenCV Zoo |
| `face_recognition_sface_2021dec.onnx` | `recognize_face/models/` | OpenCV Zoo |

The lab network has no internet, so copy them onto the workstation beforehand
rather than relying on a download at run time.

`face_db/people/` is also absent by design: it holds photographs and biometric
embeddings of real people and must never be committed. It is created the first
time someone is enrolled.
