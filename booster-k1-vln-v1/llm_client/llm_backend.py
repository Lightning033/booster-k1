"""
LLM transport layer for the Booster K1 VLN pipeline.

WHY THIS FILE EXISTS
--------------------
ask_llm() (in llm_client.py) stays provider-agnostic: it builds the prompt and
the images and parses the JSON reply. It does NOT know which model service
answers. This file is that service.

Today we talk to a local LM Studio server (OpenAI-compatible, port 1234).
To swap to OpenAI - or any OpenAI-compatible endpoint (vLLM, Together, ...) -
you only touch the CONFIG below (or set the env vars). ask_llm() never changes.

The backend contract is tiny (see VLMBackend): given a system prompt, the user
text, and the labeled images, return the model's raw text reply.
"""

from __future__ import annotations

import base64
from typing import List, Protocol, Sequence, Tuple

import cv2
import numpy as np
from openai import OpenAI

# One labeled image = (label shown just before the image, BGR image array).
LabeledImage = Tuple[str, np.ndarray]


# ---------------------------------------------------------------------------
# CONFIG - the only place you change to swap providers. Edit these values
# directly (no environment variables).
# ---------------------------------------------------------------------------
# Local LM Studio (current):
#   base_url = "http://10.12.77.33:1234/v1"    (must end in /v1)
#   api_key  = anything (LM Studio ignores the value)
#   model    = the model id shown in LM Studio's server panel
#
# OpenAI (future swap): change the three values below to:
#   base_url = "https://api.openai.com/v1"
#   api_key  = "sk-..."
#   model    = "gpt-4o"
DEFAULT_BASE_URL = "http://10.12.77.33:1234/v1"
DEFAULT_API_KEY = "lm-studio"          # LM Studio ignores this; use your real key for OpenAI
DEFAULT_MODEL = "qwen/qwen3.8-27b"     # LM Studio model id
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = 120.0


class VLMBackend(Protocol):
    """Any chat backend that takes a system prompt + user text + images and
    returns the model's raw text reply. Implement this to drop in a new
    provider without touching ask_llm()."""

    def complete(self, system_prompt: str, user_text: str,
                 images: Sequence[LabeledImage]) -> str:
        """Send system prompt + user text + labeled images to the model and
        return its raw text reply.

        Interface declaration only (this is a Protocol). The real implementation
        lives in a concrete backend such as OpenAICompatBackend below.
        """
        ...


def _to_data_url(bgr: np.ndarray) -> str:
    """Encode a BGR image as a PNG base64 data URL (OpenAI image format)."""
    ok, buffer = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("Failed to PNG-encode an image for the LLM request.")
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class OpenAICompatBackend:
    """Talks to any OpenAI-compatible server (LM Studio, vLLM, OpenAI, ...).

    The request code is identical across these providers because they all speak
    the OpenAI chat format; only base_url / api_key / model differ.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed",
                 temperature: float = 0.0, timeout: float = 120.0):
        self.model = model
        self.temperature = temperature
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def complete(self, system_prompt: str, user_text: str,
                 images: Sequence[LabeledImage]) -> str:
        # User message = the text, then each image (a short label before each).
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


# Shared instance so we reuse one client across steps.
_backend: "VLMBackend | None" = None


def get_backend() -> VLMBackend:
    """Return the shared backend instance (local LM Studio by default)."""
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
    # Self-check without a server: confirm config + image encoding work.
    dummy = np.full((8, 8, 3), 128, dtype=np.uint8)
    print("data URL prefix:", _to_data_url(dummy)[:38], "...")
    print("base_url :", DEFAULT_BASE_URL)
    print("model    :", DEFAULT_MODEL)
    print("backend  :", type(get_backend()).__name__)
