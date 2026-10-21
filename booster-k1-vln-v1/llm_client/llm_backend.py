from __future__ import annotations

import base64
from typing import List, Protocol, Sequence, Tuple

import cv2
import numpy as np
from openai import OpenAI

LabeledImage = Tuple[str, np.ndarray]


DEFAULT_BASE_URL = "http://10.12.77.33:1234/v1"
DEFAULT_API_KEY = "lm-studio"
DEFAULT_MODEL = "qwen/qwen3.8-27b"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = 120.0


class VLMBackend(Protocol):

    def complete(self, system_prompt: str, user_text: str,
                 images: Sequence[LabeledImage]) -> str:
        ...


def _to_data_url(bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("Failed to PNG-encode an image for the LLM request.")
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class OpenAICompatBackend:

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed",
                 temperature: float = 0.0, timeout: float = 120.0):
        self.model = model
        self.temperature = temperature
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def complete(self, system_prompt: str, user_text: str,
                 images: Sequence[LabeledImage]) -> str:
        user_content: List[dict] = [{"type": "text", "text": user_text}]
        for label, image in images:
            if label:
                user_content.append({"type": "text", "text": f"{label}:"})
            user_content.append({
                "type": "image_url",
                "image_url": {"url": _to_data_url(image)},
            })

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""


_backend: "VLMBackend | None" = None


def get_backend() -> VLMBackend:
    global _backend
    if _backend is None:
        _backend = OpenAICompatBackend(
            base_url=DEFAULT_BASE_URL,
            model=DEFAULT_MODEL,
            api_key=DEFAULT_API_KEY,
            temperature=DEFAULT_TEMPERATURE,
            timeout=DEFAULT_TIMEOUT,
        )
    return _backend


if __name__ == "__main__":
    dummy = np.full((8, 8, 3), 128, dtype=np.uint8)
    print("data URL prefix:", _to_data_url(dummy)[:38], "...")
    print("base_url :", DEFAULT_BASE_URL)
    print("model    :", DEFAULT_MODEL)
    print("backend  :", type(get_backend()).__name__)
