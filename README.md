# booster-k1

Voice-based vision-language navigation framework for the Booster K1 EDU humanoid robot.

A spoken command is transcribed by the STT module, passed to the VLN module, and turned into robot navigation.

```
Operator speech -> STT (k1-stt-main) -> JSON over HTTP -> VLN (booster-k1-vln-v1) -> Booster K1
```

---

## Speech-to-Text (STT) module: `k1-stt-main`

Converts an operator's spoken command into structured text and delivers it to the VLN module. It runs fully locally, with no cloud services.

### Key design decisions

- **Local and offline:** uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper). No API key is needed, and audio is never sent to a cloud service (data sovereignty is a hard project constraint).
- **Batch processing with VAD:** Voice Activity Detection segments the audio into discrete commands, which suits short robot instructions.
- **Operator-side audio:** audio is captured from the operator's device, not the robot's onboard microphone array.
- **Operator console:** a local web console (Start / Stop Listening) hosted on the lab PC and restricted to the lab network.
- **Independent of the robot's stock software:** the pipeline does not use the robot's built-in cloud voice stack.

### Pipeline

1. The operator presses **Start Listening** in the web console.
2. Audio is captured and VAD isolates the spoken command.
3. faster-whisper transcribes the segment and returns a confidence score.
4. The result is sent as JSON over HTTP to the VLN module's FastAPI endpoint.

### Output to VLN (interim schema)

The VLN team asked for all available data, and they will select the fields they need. Field names are an interim contract and may change.

```json
{
  "transcript": "string",
  "confidence": 0.0,
  "timestamp": "ISO8601",
  "language": "en",
  "audio_duration_sec": 0.0
}
```

`transcript`, `confidence` and `timestamp` are the agreed minimum. Any additional fields are optional extras.

### Setup and run

<!-- TODO: confirm each step against the code in k1-stt-main before publishing -->

```bash
# 1. Clone the repository
git clone https://github.com/Lightning033/booster-k1.git
cd booster-k1/k1-stt-main

# 2. Install dependencies (confirm the requirements file name)
pip install -r requirements.txt

# 3. Start the operator console (confirm the entry point)
python <entry_point>.py
```

Then open the console in a browser on the lab network and press **Start Listening**. Set the VLN endpoint to `<VLN_ENDPOINT_URL>`.

### Testing

The STT pipeline was evaluated against the command list supplied by the VLN team. The list was used as a test set only, not for fine-tuning. Results are recorded in the project documentation bundle.

### Scope and limitations

- Speech-to-text only. Text-to-speech and robot audio output are out of scope.
- Batch processing, not continuous streaming.
- Accuracy is a supporting metric; navigation success on the VLN side is the primary measure.
- Manual stop is separate from voice: the robot can be switched to damp mode from the Booster tablet app.

---

## Vision-Language Navigation (VLN) module: `booster-k1-vln-v1`

Receives the transcribed command from STT and drives the Booster K1's navigation. See the folder for details.

---

## Team

ICT Capstone, University of Adelaide (PG-S2-43), in collaboration with the AIML Robotics Lab.
