# K1 Speech-to-Text

Local, offline speech-to-text for voice-based Vision-Language Navigation (VLN) on the Booster K1 humanoid robot.
Part of ICT Capstone PG-S2-43, AIML Robotics Lab, University of Adelaide.

An operator presses **Start Listening**, speaks a navigation command, and presses **Stop Listening**.
The command is transcribed on the local machine, checked against the approved command list, and sent to the
VLN server. The console then shows the robot executing and the result before allowing the next command.

## Features

- **Fully local and offline.** faster-whisper runs on the machine itself; no audio leaves it and no API keys are needed.
- **Operator console.** Start/Stop Listening, live mic level, transcript, per-word confidence, robot status and command history.
- **Approved commands only.** Each transcript is matched to the VLN team's command list; the approved wording is sent,
  which also corrects small transcription errors. Unclear or unlisted commands are blocked.
- **One command at a time.** After sending, the console locks until VLN reports the task completed, cancelled, or failed.
- **Push-to-talk with pre-roll and post-roll**, so first and last words are not clipped.
- **Pre-Start filter** removes speech said before Start was pressed.
- **Voice activity detection** trims silence around each command.
- **Safety gate.** Empty audio and noise are rejected and never sent.
- **Cross-platform.** The same code runs on Windows and Ubuntu; GPU is used automatically when available, with CPU fallback.

## How it connects to VLN

STT and VLN are two separate programs on the lab PC, talking over HTTP:

1. STT sends `POST /command` with `{"command_id": "...", "text": "<approved command>"}`.
2. The VLN server replies at once with `accepted` (or `busy`) and runs `main(text)` in the background.
3. STT checks `GET /status` every second until it reports `completed`, `cancelled`, or `failed`.

The real Booster K1 server is now `../booster-k1-vln-v1/vln_server.py`. It calls
the VLN pipeline and reports completion or failure after its cleanup. The file
in `integration/vln_server.py` is the original placeholder template.
See [the integration instructions](../booster-k1-vln-v1/INTEGRATION.md) for
personal-PC checks and lab-PC setup.

The STT backend also accepts `POST /api/navigation/stop` and forwards it to
VLN's `POST /stop` using the configured VLN URL. This is separate from
`POST /api/stop`, which stops microphone recording. The navigation route returns
VLN's acknowledgement; it does not unlock listening or declare cleanup complete.
An unconfirmed request returns HTTP 502 with an error message and can be retried.
Disabled VLN integration returns HTTP 409.

The **Stop Navigation** button is enabled while navigation is running. It calls
that backend route, disables repeated clicks, and shows **Stopping…** after
acknowledgement. Listening stays locked throughout `stopping`. When VLN reports
`cancelled`, the UI shows **Navigation stopped** and unlocks. A cleanup error is
shown as a failure. A failed request displays an error and allows retry. The
Space shortcut controls listening; a focused button retains its normal keyboard
action. Once stopping has been acknowledged, the 15-minute task timeout does not
unlock listening before cleanup is confirmed.

The button's JavaScript can be checked without a browser or robot using Node.js
(tested with Node 25), from this folder:

```bash
node --test tests/test_navigation_button.mjs
```

These tests simulate the DOM and HTTP responses. Visual and actual browser
interaction checks remain manual.

### Running the integration

1. On the lab PC, start the real server from inside `booster-k1-vln-v1`, using
   the working VLN environment with ROS2 sourced:
```bash
   python -m uvicorn vln_server:app --host 127.0.0.1 --port 8000 --workers 1
```
2. In another terminal, activate the STT environment. From `k1-stt-main`, start
   the console using the complete integration configuration:
```bash
   python app.py --config config-vln.yaml
```
   This enables sending to `http://127.0.0.1:8000`, keeps approved-command
   matching, and retains console/JSON logging. Microphone and model settings for
   this launch belong in `config-vln.yaml`; it does not inherit `config.yaml`.
   Plain `python app.py` uses the default configuration with VLN sending off.

3. From a third terminal, run the read-only check before sending a voice command:
```bash
   cd ../booster-k1-vln-v1
   python3 scripts/preflight_integration.py
```
   It checks both running servers, the approved list, and the stop button without
   sending a movement command. The final microphone and robot check is done on
   the lab PC with the operator present.

The six offline communication tests are in the VLN folder. They use both
repositories with simulated transcripts and navigation, and require no ROS2,
robot or speech-model download. See the linked integration instructions to run
them on the personal PC.

Previously tested end to end on 12 Oct 2026 with the placeholder: commands were sent, the console stayed locked while
"executing", and showed "Completed in 0:10"; unapproved commands and an unreachable VLN server were handled.

## Evaluation

### Accuracy (28 Sep 2026)

20 scripted navigation commands per speaker, 3 speakers (A, B, C), quiet room, laptop microphone.
Word accuracy = 1 - WER, after lowercasing and removing punctuation. Target: at least 80% per speaker.
Participants gave consent; recordings stay local and are referred to only by letter.

| Speaker | small.en (default) | medium.en |
|---|---|---|
| A | 91.1% | 93.5% |
| B | 95.2% | 96.0% |
| C | 99.2% | 100% |
| **Combined, 60 trials** | **95.2%** | **96.5%** |
| Mean processing time | 0.13 s | 0.27 s |

`small.en` is the default: medium.en is only 1.3 points better at twice the processing time and GPU memory.
Evaluation on the VLN team's commands in the lab will follow.

### Findings

- A 0.4 s post-roll fixed last-word clipping (single-speaker baseline rose from 96.0% to 99.2%).
- Approved-command matching corrects transcription errors before they reach the robot: "It's sure to keep safe
  distance" was sent as the approved "Ensure to keep safe distance" (12 Oct).
- Commands that differ only by object ("blue bag" vs "blue bottle") are kept apart: a transcript equally close to
  two approved commands is blocked rather than guessed.
- The noise filter originally rejected short, clear commands; it now requires both a high no-speech estimate and
  uncertain words (the same rule Whisper uses).
- A vocabulary hint was tested and rejected (6 Oct): it inserted words the operator never said.
- Short commands can lose their first word; the long VLN commands are less affected.
- Numbers are transcribed as digits ("2 meters").

## Setup

### Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Ubuntu (lab PC)

```bash
sudo apt install python3-venv libportaudio2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On CPU-only machines, the two `nvidia-*` packages in `requirements.txt` can be skipped.
The first run downloads the Whisper model into `models/` (about 480 MB for small.en).

## Usage

| Command | What it does |
|---|---|
| `python app.py` | Operator console at http://127.0.0.1:8765 |
| `python demo.py` | Terminal demo: Enter to start, Enter to stop |
| `python evaluate.py record --speaker a` | Record the evaluation commands for speaker A (resumable) |
| `python evaluate.py run --all` | Evaluate every speaker; report written to `outputs/eval/` |
| `python evaluate.py run --all --model medium.en` | Compare another model size |
| `uvicorn vln_server:app --app-dir integration --port 8000` | Placeholder VLN server for testing |
| `python scripts/check_env.py` | Check Python, GPU and microphone |
| `python -m pytest` | Run unit tests (15) |

## Configuration

All settings are in `config.yaml`. Relative paths resolve from the project folder.

| Setting | Purpose |
|---|---|
| `model.size` | `small.en` (default) or `medium.en` for higher accuracy |
| `audio.max_record_seconds` | Safety cap on one recording (30 s) |
| `audio.pre_roll_seconds` / `post_roll_seconds` | Audio kept before Start and after Stop |
| `matching.enabled` | Only forward commands from the approved list |
| `matching.approved_commands_file` | The approved list (`eval/vln_commands.txt`); one command per line |
| `vln.enabled` | Send approved commands to the VLN server and wait for the result |
| `vln.url` | VLN server address (`http://127.0.0.1:8000`) |
| `vln.max_task_seconds` | Safety limit for one robot task (15 min) |
| `ui.host` | `127.0.0.1` keeps the console on this machine only |

To add an approved command, add a line to `eval/vln_commands.txt` and restart the console.

## Message format

Each command is logged to `outputs/commands.jsonl` as one JSON object. Key fields:

| Field | Meaning |
|---|---|
| `transcript` | What was recognised |
| `matched_command` | The approved wording sent to VLN |
| `accepted` | `false` means it was not sent |
| `reject_reason` | `no_speech`, `likely_noise`, `not_approved`, `ambiguous_command`, ... |
| `confidence` | 0 to 1, from the model's average token log-probability |
| `command_id` | Unique ID, shared with VLN for tracing |
| `words` | Each word with timing and confidence |

Run `python -m stt.schema` to print a complete example.

## Project structure

```
app.py                     operator console entry point
demo.py                    terminal demo
evaluate.py                multi-speaker accuracy evaluation
config.yaml                all settings
eval/commands.txt          evaluation commands
eval/vln_commands.txt      approved commands from the VLN team
integration/vln_server.py  VLN server template (FastAPI) for the VLN team
stt/
  config.py                loads config.yaml
  runtime.py               GPU/CPU selection, CUDA library setup
  audio.py                 push-to-talk recording with pre-roll and post-roll
  transcriber.py           faster-whisper, VAD, pre-Start filter, confidence, safety gate
  matcher.py               approved-command matching
  vln_client.py            sends commands to VLN and reads its status
  schema.py                JSON message format
  sinks.py                 console, JSON log, HTTP, ROS 2 outputs
  pipeline.py              connects the modules; loads the model once
  server.py                console web server, robot status, API
  web/index.html           console page
tests/                     unit tests: pre-Start filter, safety gate, command matching
scripts/                   environment check, mock receivers, ROS 2 listener
```

## Known limitations

- Not yet tested on the lab PC or with the real VLN `main()`; lab testing is planned before demo week.
- Accuracy evaluated with three speakers in a quiet room; lab-condition testing pending.
- A transcription error that turns one approved command into another (e.g. "bag" heard as "bottle") cannot be caught.
- Voice is not used to stop the robot mid-task; the tablet's damp mode is the emergency stop.
- The console has no login; it is bound to `127.0.0.1` so only the local machine can use it.
- English only, one command per recording.
- The ROS 2 output is kept for future use and has not been tested on hardware.
