# K1 Person Follow

**Vision subsystem for the Booster K1 EDU humanoid: pick one person out of a room, learn their name the first time, recognise them on every later run, and keep the robot's head pointed at them.**

Part of ICT Capstone **PG-S2-43**, *Voice Based Vision-Language Navigation for the Booster K1 Robot*, AIML Robotics Lab, Adelaide University.
Author and maintainer of this module: **Harsh Kansara**.

| | |
|---|---|
| **Detection** | [YOLOv8n](https://github.com/ultralytics/ultralytics) with ByteTrack, person class only |
| **Recognition** | OpenCV [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) detection + [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) 128-d embeddings |
| **Runs on** | Ubuntu 22.04 lab workstation, Python 3.10, ROS 2 Humble; robot is a Booster K1 EDU with a RealSense D455 |
| **Talks to** | The robot over the Booster SDK (DDS); no network services, no internet |

> **The robot does not walk.** No locomotion command is sent at any point, not even a zero one. A test asserts this.

---

## Contents

1. [What it does](#1-what-it-does)
2. [Key design decisions](#2-key-design-decisions)
3. [Architecture](#3-architecture)
4. [How one frame is processed](#4-how-one-frame-is-processed)
5. [Where it fits with STT and VLN](#5-where-it-fits-with-stt-and-vln)
6. [Results](#6-results)
7. [Setup](#7-setup)
8. [Running](#8-running)
9. [Configuration](#9-configuration)
10. [Enrolled people and the face database](#10-enrolled-people-and-the-face-database)
11. [Testing](#11-testing)
12. [Troubleshooting](#12-troubleshooting)
13. [Project structure](#13-project-structure)
14. [Development history](#14-development-history)
15. [External resources and libraries](#15-external-resources-and-libraries)
16. [Privacy, safety and ethics](#16-privacy-safety-and-ethics)
17. [Known limitations and future work](#17-known-limitations-and-future-work)
18. [Handover notes](#18-handover-notes)

---

## 1. What it does

An operator starts the pipeline from one command. The robot finds every person in the camera frame, locks onto one of them, asks for their name the first time it sees them, and from then on keeps its head pointed at that person and greets them by name on every later run.

**Features**

- **Locks onto one person, not whoever is nearest right now.** The lock is a face identity, so it survives the tracker renumbering a person who was briefly occluded.
- **Learns a name once.** The first time an unrecognised face is close enough, the terminal asks for a name and stores it with a person id (`P001`, `P002`, …) that is assigned once and never reused.
- **Head only.** No locomotion, and the arm is disabled by default.
- **No depth.** The camera rides on the head, so the pixel error is already the head angle. Distance, where it is needed at all, comes from face width and the focal length.
- **Fully offline.** No internet on the lab network; every model file is on disk before the run starts.
- **Two live views on localhost.** A detector view showing boxes, the lock and the recognised name, and a network view showing YOLOv8n's layers firing as it decides.
- **Degrades instead of failing.** A repeated frame is skipped, a frame older than 600 ms produces no command at all, an unreachable robot still runs the whole vision half under `--dry-run`.
- **242 offline tests.** Every one runs with no robot and no camera attached.

---

## 2. Key design decisions

- **ROS 2 and the Booster SDK never share a process.** Sharing one loses camera frames; on the previous project depth frames stopped arriving entirely while colour kept working, with no error anywhere. The camera process imports `rclpy` and never the SDK; the pipeline process imports the SDK and never `rclpy`. The split also enforces the SDK's one-initialisation-per-process rule, which a second call turns into a segfault.
- **The lock is an identity, not a tracker id.** ByteTrack issues a new track id after a single dropped detection, which froze the head for five seconds with the person in plain view. The face is now the thing being followed and the track id is a per-frame shortcut.
- **No depth in the loop.** The camera rides on the head, so image error converts straight into a head angle through the focal length. No depth, no world frame, no calibration rig — and half the bandwidth on a link that is already the bottleneck.
- **Lock first, ask second.** The head starts tracking on the first frame; identification happens afterwards, so nothing waits on a person typing, and the system does not interrogate everyone who walks past the door.
- **The camera topic is discovered, not assumed.** On this robot the compressed topic advertises at 30 Hz and delivers nothing to a subscriber, so the node probes, finds no frames and falls back to raw by itself.
- **The arm is opt-in.** Its torso-frame axes are not verified on this robot and a sign error sends the limb the wrong way.
- **Proportional control only.** With the camera in the loop the plant is close to an integrator, so a P term alone is stable and settles without steady-state error. A derivative term at this frame rate would mostly amplify detector jitter, and an integral term would wind up during the deliberate freezes on stale frames.

---

## 3. Architecture

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

| Module | Responsibility |
|---|---|
| `camera_process/camera_node.py` | ROS 2 node: discovers the camera topic and its QoS, converts frames, handles row padding and the RGB/BGR swap |
| `camera_process/camera_process.py` | Spawns the node in its own process and ships JPEG frames over a `multiprocessing` pipe |
| `detect_person/detect_person.py` | YOLOv8n with ByteTrack; person class only, confidence 0.30, input letterboxed to 640 |
| `recognize_face/recognize_face.py` | YuNet face detection at confidence 0.80, SFace 128-d embedding, landmark alignment |
| `face_db/face_db.py` | Enrolment, cosine matching with a threshold and a runner-up margin, person ids |
| `face_db/calibrate.py` | Measures the real same-person and different-person score distributions and suggests a threshold |
| `follow_controller/follow_controller.py` | The control law: pixel error to head angle, gains, deadband, staleness. Imports no SDK, so it is fully testable offline |
| `head_control/head_control.py` | `RotateHead` stepping, rate limiting and the clamps; sign-convention test modes |
| `arm_control/arm_control.py`, `frame_probe.py` | Hand pointing (opt-in) and a read-only transform probe |
| `sdk_client/sdk_client.py` | Mode reading and switching; tolerant parsing of the SDK's mode response |
| `live_view/live_view.py` | MJPEG page: boxes, lock state, recognised name, frame rate and frame age |
| `net_view/net_view.py` | The detector graph: forward hooks on all 23 stages, published as a live page |
| `pipeline/person_follow.py` | The loop that joins all of the above, plus the lock state machine and the enrolment prompt |
| `tools/link_sdk.py` | Finds the SDK, checks its ABI tag against the interpreter and symlinks it into the active venv |

The camera process and the pipeline process are separate programs sharing only a pipe, so either can be developed, tested and replaced on its own.

---

## 4. How one frame is processed

| Step | What happens | Why |
|---|---|---|
| 1. Capture | A 1280×720 colour frame crosses the pipe as JPEG | ROS 2 and the SDK cannot share a process |
| 2. Detect | YOLOv8n + ByteTrack: boxes, track ids, person class only | Confidence 0.30 rather than the default 0.25, which made detections flicker and made the tracker renumber people |
| 3. Choose | The locked track if still present, otherwise the nearest face by apparent width; the three nearest are checked for a known face before an unknown one is asked about | Someone walking between the robot and the locked person must not steal the lock |
| 4. Recognise | YuNet finds the face, SFace turns it into 128 numbers, compared by cosine similarity | Faces under 40 px are tracked but never matched — a small face gives an unreliable embedding |
| 5. Aim | `atan2(cx − px, fx)` gives the yaw error directly; the aim point is the face, or the top sixth of the body box when there is no face | The centre of a body box is a stomach; aiming there makes the robot stare at a torso |
| 6. Move | One capped `RotateHead(pitch, yaw)` step — radians, absolute | Errors under the deadband, and frames older than 600 ms, produce no command at all |

---

## 5. Where it fits with STT and VLN

This module answers a question the rest of the system needs: **who is the robot interacting with.** Speech-to-text hears a command but not who said it; VLN moves the robot but does not decide who it is for.

| Module | Owner | Role |
|---|---|---|
| [`k1-stt-main`](../k1-stt-main) | Dedeepya Korukonda | Hears and validates a spoken command |
| [`booster-k1-vln-v1`](../booster-k1-vln-v1) | Aditya | Turns an approved command into navigation |
| `k1-person-follow` | Harsh Kansara | Decides who the robot is attending to and keeps the head on them |

**Not wired together in this release, and deliberately so.** Person follow and the STT/VLN pair both want the camera and both want the SDK, and the one-init-per-process rule means they cannot naively share a process. The architecture here is the right shape for the integration — separate processes talking over a pipe — but arbitrating the head between *following a person* and *navigating* is real design work that was out of scope this semester. The interface when it happens is the lock state: a person id and a name.

---

## 6. Results

### 6.1 Measured on the robot, with the real models on real frames

| Figure | Value |
|---|---|
| Loop frame rate over WiFi | 4–5 Hz on the raw topic |
| Time to centre a person 30° off axis | ~1.5 s at that frame rate |
| Settling time at 10 Hz (for comparison) | ~0.7 s, no overshoot |
| Worst same-person score vs best impostor | 0.79 vs 0.02 |
| Recognition range at the default detect width | 3.6 m |
| Enrolment range (face ≥ 54 px) | ~1.9 m |
| Detector graph overhead, before throttling | 9.2 ms/frame (11% of the loop budget) |
| Detector graph overhead, after throttling to 0.28 s | within measurement noise |

Frame **age** is the number to watch at run time, not frame rate: under 300 ms is healthy, and an age that climbs means frames have stopped arriving.

The frame rate is the weakest number here and the reason the gain is conservative. The compressed camera topic advertises but delivers nothing to a subscriber on this robot, so the pipeline falls back to the raw topic over WiFi. Fixing that should roughly double the rate and allow the gain to come back up.

### 6.2 Tuning the control law

`KP_YAW = 0.55` is the single most important constant in the module. Because the camera rides on the head, an error measured in the image *is* roughly the angle the head must turn, and once it turns the next frame already shows a smaller error. At gain 1.0 the head tries to cancel the whole error in one step, arrives a frame late and oscillates. Pitch is slightly lower at 0.45 because the vertical travel is more limited.

### 6.3 What testing changed

| Problem found | Fix | Effect |
|---|---|---|
| A dropped detection renumbered the person | Lock on face identity, not track id | Head no longer freezes for five seconds on a vanished track |
| Mode read from the wrong field of the SDK response | Tolerant field parsing with a `None` fallback | Loop no longer refuses to start while the robot is correctly in walk mode |
| One face saved under two names | Enrolment detects it and adds the view to the existing person | Recognition no longer ends silently for that person |
| `--speed 0` ran the head at full speed | Explicit `None` check instead of `value or default` | The most dangerous possible misreading of that flag is gone |
| A visitor scored 0.41 against an enrolled colleague | Threshold 0.40 → 0.55, two-frame agreement, 40 px recognition floor | Wrong name no longer attached with confidence |
| Enrolment prompt shredded by status lines | Console output buffered while a name is typed | The prompt is readable and survives re-acquisition |

None of those six raises an exception. Each produces plausible behaviour that is wrong, which is exactly what a live demonstration cannot distinguish from working.

---

## 7. Setup

### What you need

**Hardware.** The K1 powered on with its ROS 2 camera stack running, the lab workstation on the robot's hotspot network, and a stand or a spotter — a mode change can make the robot shift its feet.

**Network.** The robot answers at `192.168.5.3`. Its interface is `wlP5p1s0`, the workstation's is `wlp5s0`. There is no internet on this network, so everything must be on disk before starting — no model downloads at run time.

**Software.**

| Needs | Where it comes from |
|---|---|
| ROS 2 Humble | `/opt/ros/humble` |
| Python 3.10 | system |
| `booster_robotics_sdk_python` | installed in `k1_env` |
| ultralytics, OpenCV, numpy < 2, lap | installed in `k1_follow_env` |
| YOLOv8n weights | `detect_person/yolov8n.pt` |
| YuNet and SFace ONNX models | `recognize_face/models/` |

**Why two environments.** `k1_env` holds the robot SDK. `k1_follow_env` holds the vision stack and pins numpy below 2, because ROS 2 and OpenCV 4.11 break against numpy 2. Rather than merge them and risk one breaking the other, the SDK is symlinked from the first into the second by `tools/link_sdk.py`.

### One-time setup

```bash
# 1. the vision environment — --system-site-packages is required,
#    or it cannot see ROS2 and the camera process will not run
python3 -m venv --system-site-packages ~/k1_follow_env
source ~/k1_follow_env/bin/activate
pip install -r requirements.txt

# 2. link the SDK in (once per environment)
python tools/link_sdk.py          # prints VERIFIED on success

# 3. confirm the models are present
ls detect_person/yolov8n.pt
ls recognize_face/models/*.onnx
```

`link_sdk.py` finds `booster_robotics_sdk_python` wherever it is installed, checks it was built for the same Python version, symlinks that one module into the active environment and verifies the import in a fresh interpreter. Nothing is copied or moved; `--unlink` reverses it. If the SDK was built for a different Python version it refuses to link — a compiled extension cannot cross versions.

### Every session

```bash
cd <project root>
source /opt/ros/humble/setup.bash
source ~/k1_follow_env/bin/activate
export K1_NET_INTERFACE=wlp5s0
```

Source ROS 2 **before** activating the environment. The other order leaves ROS 2's Python ahead of the environment's on the path and the wrong numpy is imported.

Run everything from the project root — the modules import each other by relative path.

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

Step 2 expects `Ran 242 tests` and `OK`. Filter for the summary rather than using `tail`: the tests print to stdout and the summary to stderr.

In step 3, watch the **frame age**, not the frame rate. Under 300 ms is healthy; an age that climbs means frames have stopped arriving.

In step 4 it must **name** a mode such as `kWalking`. If it prints an object instead, the mode could not be read and the follow loop will refuse to start. Head commands are rejected with error 400 in any other mode.

---

## 8. Running

```bash
# rehearsal — nothing moves, robot can be off
python pipeline/person_follow.py --lock-nearest --dry-run --window --port 8090

# the normal run
python pipeline/person_follow.py --lock-nearest --window --port 8090

# with the detector graph on a second port
python pipeline/person_follow.py --lock-nearest --net-view --window --port 8090
```

Open `localhost:8090` for the camera view and `localhost:8091` for the detector graph. `Ctrl+C` exits cleanly — the head recentres, the arm lowers, the camera closes. In the desktop window: `q` quit, `c` recentre, `u` unlock.

Dry run touches no SDK at all, and enrolment still works, so a person can be registered before the robot is switched on. It is also the fallback if the robot is uncooperative on the day.

| Flag | What it does |
|---|---|
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

## 9. Configuration

Environment variables, so nothing needs editing to move machines. Only `K1_NET_INTERFACE` normally has to be set.

| Variable | Default | What it is |
|---|---|---|
| `K1_NET_INTERFACE` | `wlp5s0` | Interface that reaches the robot |
| `K1_DDS_DOMAIN` | `0` | DDS domain id |
| `K1_FACE_THRESHOLD` | `0.55` | Cosine score a match must reach |
| `K1_FACE_MARGIN` | `0.05` | Margin over the runner-up |
| `K1_FACE_DB` | `face_db/people` | Where enrolled people are stored |
| `K1_FACE_MODELS` | `recognize_face/models` | Where the ONNX files are |
| `K1_FACE_DETECT_WIDTH` | `960` | Face-detection width; higher sees further |
| `K1_YOLO_MODEL` | `detect_person/yolov8n.pt` | YOLO weights |

Constants that are not environment variables, because changing them needs the reasoning in the source: `KP_YAW = 0.55`, `KP_PITCH = 0.45`, `DEADBAND_RAD = 0.02` (about 1.1°), `STALE_FRAME_S = 0.60`, `YAW_LIMIT = 0.90 rad`, `PITCH_LIMIT = 0.35 rad`.

Detection width sets how far away a person can be recognised. Measured on a 1280×720 frame with fx = 644:

| Width | Smallest face | Usable range | Cost per frame |
|---|---|---|---|
| 640 | 39 px | 2.6 m | 10 ms |
| 960 (default) | 29 px | 3.6 m | 21 ms |
| 1280 | 23 px | 4.5 m | 36 ms |

---

## 10. Enrolled people and the face database

The robot locks first and asks second. Once the face is wide enough to store, a prompt appears in the terminal and on the page; typing a name saves the face with an ID (`P001`, `P002`, …) assigned once and never reused.

Enrolment needs the face about **54 px wide, so inside roughly 1.9 m**. Further away it still follows the person — it just will not save a sample it cannot trust, because that sample is what every later run compares against.

```
face_db/people/<name>/
  meta.json          name, slug, ID, timestamp, sample count
  embeddings.npy     (N, 128) float32, one row per enrolled sample, up to 12
  sample_01.jpg      the frame each sample came from
```

```bash
python face_db/face_db.py list
python face_db/face_db.py whois photo.jpg
python face_db/face_db.py remove harsh
python face_db/calibrate.py --pairs       # re-measure the threshold
```

**The threshold.** A match needs a cosine score of at least **0.55** with a **0.05** margin over the runner-up, and the same identity has to win on **two consecutive frames**. 0.40 was tried first and proved too loose: a visitor scored 0.41 against an enrolled colleague and was briefly named as them.

The root cause is worth stating, because it is not obvious. The margin test compares the best score against the runner-up — but with only one person enrolled the runner-up is 0.0, so the margin test passes trivially, every time. The one check that should have caught a weak match was structurally disabled by the size of the database.

`calibrate.py` re-measures both distributions on whichever faces are enrolled and places the suggested threshold just under the true-match floor, **not** at the midpoint between the two distributions — the midpoint would have *lowered* the threshold to around 0.41.

**One face, one name.** Saving the same person under two names makes every later match ambiguous and silently ends recognition for them. The code detects this and adds the new view to the existing person instead.

---

## 11. Testing

| Suite | Command | What it covers |
|---|---|---|
| Offline tests (242) | `cd tests && python -m unittest discover -s .` | Everything below |
| SDK link check | `python tools/link_sdk.py` | The SDK imports in a fresh interpreter |
| Arm frame probe | `python arm_control/frame_probe.py` | Reads transforms off the robot; commands nothing |

All 242 tests run with no robot and no camera attached. They cover:

- the control law's sign conventions, convergence and clamps
- the stale-frame freeze and the deadband
- the face database's threshold and margin gates, including the single-person case that caused the false match
- the camera pipe protocol, with real spawned processes
- the ROS conversions, including row padding and the RGB/BGR swap
- the identity re-acquisition path when a track id changes under a locked person
- the arm's geometry, hand choice, rate limit, resting on shutdown, and that a refused pose switches the arm off and leaves the head following
- a bait trap proving no locomotion command is ever sent

Roughly 3,335 lines of test against 4,861 lines of module code.

**Not covered by tests:** anything needing the robot or a camera — actual head motion, the real frame rate, topic discovery against a live ROS graph, and the arm. The tests check the *logic*, not the *choice of constant*: no test tells you the recognition threshold is right for a larger group of people.

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: booster_robotics_sdk_python` | Wrong environment, or never linked | Activate `k1_follow_env`, then `python tools/link_sdk.py` |
| `ModuleNotFoundError: rclpy` | Environment built without `--system-site-packages` | Rebuild it with that flag |
| numpy errors that never mention numpy | numpy 2 against ROS 2 or OpenCV 4.11 | `pip install "numpy<2"` |
| `Address already in use` | A previous run still holds the port | `--port 8091`, or `pkill -f person_follow.py` |
| Camera reports no frames | The node lists every topic it tried | Read that report; it falls back on its own |
| Frame age climbing steadily | Frames have stopped arriving | Restart the camera process; check the WiFi link |
| Stuck on `searching` | Console names the reason, usually a face too small | Step closer; it prints the measured distance |
| Head does not move | `--dry-run` set, or not in walk mode | Drop the flag; re-run `sdk_client.py walk` |
| Error 400 on a head command | Robot left walk mode | `python sdk_client/sdk_client.py walk` |
| Head motion looks abrupt | Default speed too quick for the room | `--speed 0.3` |
| Wrong person recognised | Threshold too loose for these faces | Enrol everyone present, then `calibrate.py --pairs` |
| Robot unreachable | Hotspot dropped, or the K1 is asleep | `--dry-run`; the whole vision half still demonstrates |

None of these needs a code change — each is a flag, a reconnect, or a step forward.

---

## 13. Project structure

```
pipeline/person_follow.py       the loop, lock state machine, enrolment prompt
camera_process/
  camera_node.py                ROS 2 node: topic and QoS discovery, conversions
  camera_process.py             spawns the node, ships JPEG over a pipe
detect_person/detect_person.py  YOLOv8n + ByteTrack, person class only
recognize_face/recognize_face.py  YuNet detection, SFace embedding, alignment
face_db/
  face_db.py                    enrolment, matching, person ids
  calibrate.py                  measures score distributions, suggests a threshold
  people/                       enrolled people — gitignored, never committed
follow_controller/follow_controller.py  the control law; imports no SDK
head_control/head_control.py    RotateHead stepping, rate limit, clamps
arm_control/
  arm_control.py                hand pointing (opt-in)
  frame_probe.py                read-only transform probe
sdk_client/sdk_client.py        mode reading and switching
live_view/live_view.py          MJPEG page
net_view/net_view.py            detector graph page
tools/link_sdk.py               finds and symlinks the SDK into the active venv
tests/                          242 offline tests
requirements.txt                pip packages (ROS 2 and the SDK are not pip-installable)
```

---

## 14. Development history

In order. Dates are in the team repository's commit history.

| Stage | Work |
|---|---|
| 1 | Camera process and ROS 2 node; topic and QoS discovery with fallback to raw |
| 2 | Process separation after frames went missing with ROS 2 and the SDK in one interpreter |
| 3 | YOLOv8n + ByteTrack detection; MJPEG live view |
| 4 | Control law: pixel error to head angle, gains, deadband, staleness; head sign calibration on the robot |
| 5 | `tools/link_sdk.py` — SDK discovery, ABI check and symlinking, after environment contamination cost a day |
| 6 | Face recognition: YuNet + SFace, the face database, enrolment and person ids |
| 7 | Identity lock, replacing the tracker-id lock that froze the head |
| 8 | Multi-person locking: the three nearest candidates are checked for a known face |
| 9 | Head speed control and hold-the-lock behaviour |
| 10 | Mode-parsing fix after the loop refused to start on a correctly walking robot |
| 11 | Name-prompt fixes: buffered console output, prompt survives re-acquisition |
| 12 | Recognition hardening after a false match: threshold 0.55, two-frame agreement, 40 px floor, `calibrate.py` |
| 13 | `net_view` detector graph, with the publish throttle that made it free |
| 14 | Arm control and the read-only frame probe (opt-in, unverified) |
| 15 | Documentation: this README and the setup guide |

---

## 15. External resources and libraries

| Resource | Used for | Licence |
|---|---|---|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Person detection and the ByteTrack implementation | **AGPL-3.0** |
| `yolov8n.pt` | Detection weights, COCO 80 classes, person = class 0 | AGPL-3.0 (Ultralytics) |
| [ByteTrack](https://github.com/ifzhang/ByteTrack) | Multi-object tracking algorithm (used via ultralytics) | MIT |
| [OpenCV](https://opencv.org/) (`opencv-python`) | Image handling, `FaceDetectorYN`, `FaceRecognizerSF`, the desktop window | Apache-2.0 |
| [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) `face_detection_yunet_2023mar.onnx` | Face detection (228 KB) | MIT |
| [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) `face_recognition_sface_2021dec.onnx` | 128-d face embeddings (37 MB) | Apache-2.0 |
| [NumPy](https://numpy.org/) | Frame and embedding arrays; pinned below 2 | BSD-3-Clause |
| [lap](https://github.com/gatagat/lap) | Linear assignment for ByteTrack; pinned so it is installed in advance | BSD-2-Clause |
| [ROS 2 Humble](https://docs.ros.org/en/humble/) (`rclpy`, `sensor_msgs`) | Camera topic subscription | Apache-2.0 |
| `booster_robotics_sdk_python` 1.3.6 | Robot control (DDS) | Vendor licence, Booster Robotics |

> **Licensing note.** Ultralytics YOLOv8 is AGPL-3.0, which requires derivative works that are distributed or offered over a network to be released under the same licence. That is appropriate for an open academic project, but anyone reusing this module commercially needs either an Ultralytics enterprise licence or a different detector.

No cloud service is used, and the lab network has no internet route. Both live views use only the Python standard library (`http.server`) with plain HTML and JavaScript.

---

## 16. Privacy, safety and ethics

This module stores photographs of real people and the biometric data derived from them. That is the most sensitive thing in the whole capstone and it is treated as such.

| Concern | Why it matters here | How this module responds |
|---|---|---|
| Biometric data is personal data | A 128-number embedding identifies a specific person; being a vector rather than a photograph changes nothing | Treated as personal data throughout, never as a derived feature |
| Accidental publication | The first archive prepared for upload contained both enrolled people's images and embeddings, and would have put them in a public repository | `face_db/people/*` is in `.gitignore` with a comment explaining why; the directory ships containing only a `.gitkeep`; verified before the first commit |
| Enrolment without awareness | Silent enrolment would make the robot a covert biometric recorder | Enrolment is explicit: the terminal asks for a name and the person types it |
| Retention | Data kept past its purpose | Removed with `face_db.py remove <slug>` once the demonstrations are finished |
| Data leaving the machine | Cloud face services have repeatedly leaked or been reviewed by humans | Nothing leaves the workstation; the lab network has no internet route and both views bind to localhost |
| Physical safety | A head that moves unexpectedly near a person | Head only, no locomotion, arm off by default, yaw and pitch clamped, rate limited, clean recentre on exit |

**If you reuse this module:** ask each person before enrolling them, say what it is for, keep `face_db/people/` out of every repository, shared drive and backup, and delete it when the work is done. Removing it in a later commit does not remove it from git history.

---

## 17. Known limitations and future work

**Not verified**

- `PITCH_LIMIT = 0.35 rad` is deliberately conservative. The published K1 range is wider; raise it only after measuring.
- The arm's torso-frame axes. `frame_probe.py` exists to settle them, read-only. Everything *around* the arm is tested: the geometry, the hand choice, the rate limit, resting on shutdown, and that a refused pose switches the arm off and leaves the head following.
- Why the compressed camera topic advertises but delivers nothing to a subscriber on this robot. Not blocking — the node falls back to raw — but it is the reason the frame rate is 4–5 Hz.
- Behaviour with more than three people at once; the search checks the three nearest.
- Recognition in rooms other than the one it was calibrated in. Lighting robustness has not been characterised.
- End-to-end latency as a single figure. The loop rate and the settling time were measured; the per-stage breakdown was not.

**Known not working**

The robot cannot speak. It has no audio hardware connected — no driver bound at `8-001c`, no USB-C media device and no iFlytek microphone array — and its TTS is a cloud service with no internet route. Nothing in this module depends on sound; names are shown on the page instead.

**Suggested next steps**

1. Fix the compressed camera topic. The frame rate roughly doubles and the gain can come back up.
2. Set `PITCH_LIMIT` from Booster's published range rather than the current safe guess.
3. Run `frame_probe.py` and settle the arm frame convention before anything commands an arm.
4. Enrol each person from several distances and under different lighting, then re-run `calibrate.py --pairs`, so the threshold rests on more than two people.
5. Test with more than three people in frame.
6. Define the head-arbitration contract with VLN, so following a person and navigating can run together.

---

## 18. Handover notes

- Start with `python tools/link_sdk.py`, then the offline tests. Neither needs the robot.
- `--dry-run` runs the entire vision half with the robot switched off, and enrolment still works. Use it to rehearse, and as the fallback if the robot is uncooperative on the day.
- Read `follow_controller/follow_controller.py`'s module docstring before changing any gain. The reasoning for 0.55 is written there.
- `face_db/people/` must never be committed. Check `.gitignore` is intact before any commit that touches this folder.
- Do not enable `--arm` until `frame_probe.py` has been run and the axes confirmed.
- Supervisors: Zerui Li and Xiangyu Shi. Client: A/Prof Qi Wu, AIML Robotics Lab.
