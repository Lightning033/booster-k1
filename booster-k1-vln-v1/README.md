# Booster K1 VLN v1

The lab PC runs the complete loop: capture RGB/depth, draw grids, ask the model,
map its floor point to camera coordinates, turn the body, walk, and repeat.
The robot must already be standing in PREP or WALK mode.

## Movement

Capture and motion share `VIEW_TURN` in `robot_control/robot_control.py`:
front = 0, left = +1 radian, right = -1 radian. Capture centers the head before
returning. For a valid move, the body then:

1. Turns to the selected view: left +1 radian, right -1 radian, front no turn.
2. Stops, then turns toward the selected point by `atan2(-X, Z)` radians.
3. Stops, then walks `min(sqrt(X*X + Z*Z), MAX_STEP_DISTANCE_M)` metres
   forward and stops again.

`MAX_STEP_DISTANCE_M = 1.5` in `robot_control/robot_control.py` limits the
requested walk per navigation decision. Closer targets use their shorter
distance. After each step, the pipeline captures fresh views and asks the LLM
again; it does not automatically finish the remaining distance to an old target.
At 0.2 m/s, the maximum requested walking time budget is 7.5 seconds. This is a command limit,
not a guarantee of physical stopping distance. Actual motion remains open-loop.

Camera X points right, while positive body yaw turns left. This explains the
minus sign in `atan2(-X, Z)`. Camera Y points down and is ignored for ground
distance. As requested for v1, camera mount translation and pitch are ignored;
the head yaw is handled by step 1. No TF lookup is needed.

Motion follows the tested scripts in `reference-code/`: forward speed 0.2 m/s,
yaw rate 0.5 rad/s, and a 5-second wait after entering WALK mode. Time equals
distance / speed or angle / yaw rate. Each motion ends with zero velocity and
a 0.5-second settling pause. The timer starts immediately before sending Move;
time spent waiting for its reply is subtracted from the remaining wait. If the
reply arrives after the deadline, stop is sent as soon as the call returns.
A blocked call cannot be interrupted by this timer. Stop delivery delays, sleep
overruns, and physical stopping distance remain. These are timed estimates,
without odometry feedback.

Missing, nonnumeric, nonfinite, or out-of-range model coordinates reject the
move. Coordinates must be JSON numbers from 0 through 1000; they are never
clamped to an edge. Rejected moves are saved with an internal `retry` action
and the rejected response, then fresh images and a new LLM call follow without
body movement. Retries count toward the existing 20-decision limit. Invalid
views, malformed JSON, and unknown actions still stop navigation.

Zero or otherwise unusable depth at the selected pixel skips both turning and
walking. Nearby pixels are never used to replace that reading; the pipeline
captures again for the next decision. Invalid targets or failed SDK commands abort the
run. Every timed motion attempts a stop even on Ctrl+C or an RPC exception.
The pipeline also attempts stop, waits 10 seconds after a successful stop,
and stays in WALK mode on exit. It closes the camera process even if an earlier
cleanup call fails. The standalone capture script also stops in WALK mode.
The pipeline permits at most 20 decisions and at most
two consecutive look-behind turns. There is no live obstacle checking during
a walk; v1 relies on the model selecting a clear floor path.

## Task outcome for integration

Call `pipeline.pipeline.main(instruction)` with the command text. It returns
`None` only after the model reports arrival, gesture handling finishes, and
cleanup completes. This is model-reported arrival, not an independently measured
goal position. An invalid navigation reply, the 20-decision limit, or the
look-behind limit raises `RuntimeError` with a reason. Camera, model, motion,
gesture, and cleanup errors also propagate. The existing stop and camera cleanup
are attempted on these exits. An integration server can therefore report
`completed` on a normal return and `failed` on an exception. Running the script
directly now also reports those early stops as errors (a nonzero exit).

## Gesture at the goal

When the model itself chooses `stop`, the pipeline makes one more, text-only
LLM call with the navigation instruction (`ask_gesture()` in
`llm_client/llm_client.py`). The model replies with
`{"gesture": "nothing" | "dab" | "wave" | "handshake"}`. It chooses a gesture
only when the instruction asks for one, and an unclear reply means `nothing`.
`wave` is the cheering gesture with both arms. `perform_gesture()` in
`robot_control/robot_control.py` starts the gesture, waits `GESTURE_SECONDS`
(10 seconds), and then stops it, even on Ctrl+C. The usual cleanup follows.
There is no gesture after a stop caused by an unusable model reply, after the
decision limit, or after the look-behind limit.

Test one gesture with the robot already in WALK mode (it stays in WALK):

```bash
python booster-k1-vln-v1/robot_control/robot_control.py dab   # or wave, handshake
```

Check the gesture choice against LM Studio without the robot:

```bash
python booster-k1-vln-v1/llm_client/llm_client_testing/test_ask_gesture.py
```

## Run on the lab PC

Use the existing environment with ROS2 sourced, the Booster Python SDK,
NumPy, OpenCV, and `openai` installed. Start LM Studio with a vision model.

- Check `_INTERFACE` in `robot_control/robot_control.py` (currently `wlp5s0`).
- Camera capture runs in its own process (`camera_node.py`), which loads ROS2
  but never the Booster SDK. With both in one process, depth frames went missing
  on the lab PC, even with ROS2 started first. The camera process starts before
  the SDK and must receive one RGB and one depth frame before the robot gets any
  command. After each head move, the pipeline waits 1 second, then asks for one
  fresh pair; frames queued during the move are discarded. One request runs at
  a time, every wait has a time limit, and Ctrl+C is handled by the pipeline.
- Check `DEFAULT_BASE_URL` and `DEFAULT_MODEL` in `llm_client/llm_backend.py`
  against the lab PC's LM Studio server. These are file settings, not env vars.
- Capture uses the recorded 1280x720 color intrinsics and depth aligned to color,
  in metres. ROS `16UC1` depth is converted from millimetres.
  Before calling the LLM, the pipeline checks all three RGB views and available
  depth arrays against 1280x720. A wrong resolution stops navigation with an
  error before any body turn or walk. Missing depth still skips its movement target.

From the repository root:

```bash
python booster-k1-vln-v1/pipeline/pipeline.py "Move straight to leave the room, then turn left and stand near the human"
```

Omit the instruction argument to use `NAVIGATION_INSTRUCTION` in the pipeline.
Each decision saves RGB, grid images, and depth under
`checkpoints/run-N/checkpoint-M/`. After each LLM call, `llm_output.txt` in that
same folder stores the complete parsed `result` as indented JSON: thought,
action, view, u/v, and new_planning. It is saved before executing the action,
including `stop` and `look_behind` decisions.
The console prints the selected pixel, camera XYZ, turn angles, distance, and
command durations. Press Ctrl+C to request stop and cleanup.

## HTTP server for STT

See [INTEGRATION.md](INTEGRATION.md) for personal-PC checks, transfer instructions,
and the two-terminal lab startup using STT's `config-vln.yaml`.

`vln_server.py` wraps the existing `main(instruction)` function. From inside
`booster-k1-vln-v1`, with the existing VLN environment activated and ROS2 sourced:

```bash
python -m pip install -r requirements-server.txt
python -m uvicorn vln_server:app --host 127.0.0.1 --port 8000 --workers 1
```

Starting the server does not start navigation. Check its status without moving
the robot:

```bash
curl http://127.0.0.1:8000/status
```

It initially reports `status: "idle"`. STT can send `POST /command` with
`{"command_id": "voice-1", "text": "<navigation instruction>"}`. This endpoint
starts real navigation: it returns `accepted` immediately and launches
`vln_worker.py` in a separate Python process. That wrapper calls `pipeline.main`
with the instruction and reports the result after cleanup. A background thread
in the server monitors the process. While a task is running,
another command receives `busy` with the active command ID; it is not queued.
Blank/non-string instructions receive HTTP 422. If no command ID is supplied,
the server generates one. The same ID appears in replies, status and server logs.

`GET /status` returns `status`, `command_id`, `text`, `elapsed_s`, and `message`.
Status is `idle`, `running`, `stopping`, `completed`, `cancelled`, or `failed`; failure messages include
the pipeline error. Completion is reported only after `main` returns, including
its final stop, settling wait and camera cleanup. State describes the latest task
and is held in memory; restarting the server resets it. Command IDs are for
tracking, not persistent duplicate detection.

Each task uses the server's Python interpreter and inherited environment, with
the VLN project as its working directory. Navigation output remains visible in
the server terminal. The wrapper writes its outcome to a temporary JSON file;
the server reads it after the process exits and removes the temporary directory.
A process that exits without a valid result is reported as failed.

Run exactly one server process with one worker and no `--reload`. The busy lock
is local to that process. The server owns robot commands while it is running;
do not run the standalone pipeline alongside it. The STT approved-command check
remains on the STT side; the server accepts any nonempty instruction text.

To stop the active navigation run while keeping the server running:

```bash
curl -X POST http://127.0.0.1:8000/stop
curl http://127.0.0.1:8000/status
```

`POST /stop` returns `stopping` with the active command ID. The monitor sends one
SIGINT after the child installs its interrupt handler. Navigation unwinds through
its existing cleanup; further stop requests cannot interrupt that cleanup.
Commands remain `busy` until the child exits. The final status is `cancelled`,
or `failed` if cleanup raises an error. If completion wins the race with stopping,
the status may remain `completed`. An idle or finished run is unchanged by `/stop`.
Cancellation requests during final pipeline cleanup let that cleanup finish.

The STT UI has a Stop Navigation button that forwards the request through its
backend. It shows Stopping… after acknowledgement and allows retry on request
errors. STT polling now recognizes `stopping`, `cancelled`, and cleanup failures.
Listening remains locked through cleanup, then the UI shows Navigation stopped
and unlocks. Once stopping is acknowledged, the task timeout does not unlock
listening before VLN reports a final result.

Normal server shutdown rejects further submissions and waits for the current
run to finish, including its cleanup. **Ctrl+C on this server is not a navigation
cancel command:** the current task continues while shutdown waits. A blocked
SDK call can keep shutdown waiting. Forced process termination cannot guarantee
cleanup. Navigation runs in its own process session; terminal interrupts are
not yet forwarded to it. Automatic cancellation on server shutdown is separate
from the UI stop workflow and is not implemented yet.
`/stop` is a software cancellation request, not a hardware emergency stop;
blocked native SDK calls can delay interruption and physical stopping requires
working communication with the robot.

STT sending is enabled in `k1-stt-main/config-vln.yaml`, selected with
`python app.py --config config-vln.yaml`; its default configuration still runs
without VLN sending. Complete the offline checks below on the personal PC,
then start both programs on the lab PC. The worker and shutdown behavior follow the
[FastAPI lifespan API](https://fastapi.tiangolo.com/advanced/events/) and
[Python thread lifecycle](https://docs.python.org/3/library/threading.html#thread-objects).

## Offline development tests

The server tests additionally need its dependencies and HTTPX. Install into
your development environment, from inside `booster-k1-vln-v1`:

```bash
python -m pip install -r requirements-server.txt 'httpx>=0.27,<1' 'pyyaml>=6,<7'
```

Then run the suite from the parent repository root:

```bash
conda run -n aiml-k1 python -m unittest discover -s booster-k1-vln-v1/tests -v
```

Tests use the saved RGB/depth files and fake SDK/ROS boundaries. They exercise
the real pipeline, coordinate conversion, motion commands, and failure cleanup.
Camera-process tests start real stand-in processes without ROS2.
Server tests inject fake navigation and use FastAPI's in-process HTTP test client
to check command validation, busy handling, task outcomes and shutdown.
Navigation-process tests also launch real Python child processes with simulated
navigation. They check process isolation, exact instruction delivery, cleanup,
error reporting, abnormal exits and subsequent commands.
They also verify cancellation during execution and startup, repeated stop
requests, protected cleanup, and cleanup failure reporting.
Keep `k1-stt-main` alongside this folder for the integration tests. These use
the real STT approved-command matcher, sending/polling code, console state, and
VLN HTTP routes. Simulated transcripts replace microphone/Whisper input; fake
navigation replaces hardware. Requests are routed in process, so they do not
test a real Uvicorn process or network sockets. No ROS2 installation is needed.

Run just the communication tests on the personal PC, from the parent folder:

```bash
python -m unittest discover -s booster-k1-vln-v1/tests -p test_stt_vln_integration.py -v
```

They verify successful completion, failure messages, a fresh command after
failure, approved wording, rejected commands, busy handling, and the STT timeout.
They check the status data that the UI renders; browser rendering and actual
microphone/robot behavior still need a lab test.

They do not connect to the robot or LM Studio. No pytest install is required.
Passing these tests does not verify physical turn angles or walking distances;
those still need testing on the lab robot.

## Sources

- Tested local commands: `reference-code/move_ahead.py` and `reference-code/turn_body.py`.
- Recorded calibration: `documentation-reference/booster-k1-camera-details.txt`.
- [Booster motion API](https://docs.booster.tech/docs/developer-guide/cpp/rpc/motion/):
  `Move(vx, vy, vyaw)` uses m/s and rad/s; `RotateHead` uses radians.
- [SDK source](https://github.com/BoosterRobotics/booster_robotics_sdk/blob/main/include/booster/robot/b1/b1_loco_client.hpp):
  zero means success; other return codes indicate errors.
- [ROS coordinate conventions](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst):
  camera optical axes are X right, Y down, Z forward.
