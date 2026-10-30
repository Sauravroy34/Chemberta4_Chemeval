"""
Multimodal model interface for ChemEval.

Supports vision-capable LLMs via litellm (GPT-4o, Claude, Gemini, etc.).
Images are sent inline as base64-encoded data URIs alongside the text query.
"""
import os
import io
import re
import time
import base64
from dotenv import load_dotenv
import litellm
from litellm import completion_cost

load_dotenv(override=True)
litellm.drop_params = True

LANGFUSE = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    from langfuse import get_client
    LANGFUSE = get_client()
else:
    print("WARNING: Langfuse keys not set - nothing will be logged to Langfuse")


def pil_to_base64(img, fmt="PNG"):
    """Convert a PIL Image to a base64 data-URI string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def build_content_parts(query, image):
    """
    Build a list of content parts for a multimodal message.

    The query may contain one or more <ImageHere> placeholders.
    If only one image is provided, it is inserted at the first placeholder
    (or appended if no placeholder exists).

    Returns a list of {"type": "text", "text": ...} and
    {"type": "image_url", "image_url": {"url": ...}} dicts.
    """
    parts = []
    img_uri = pil_to_base64(image)

    # Split on <ImageHere> to interleave text and image
    segments = re.split(r"<ImageHere>", query)

    for i, seg in enumerate(segments):
        text = seg.strip()
        if text:
            parts.append({"type": "text", "text": text})
        # Insert the image after the first text segment (at the first placeholder)
        if i == 0:
            parts.append({
                "type": "image_url",
                "image_url": {"url": img_uri},
            })

    # If no text was added (shouldn't happen), at least add the image
    if not parts:
        parts.append({
            "type": "image_url",
            "image_url": {"url": img_uri},
        })

    return parts


class MultimodalModel:
    """Vision LLM via litellm (GPT-4o, Claude 3.5, Gemini 1.5, etc.)."""

    def __init__(self, model_name, max_retries=6, **kwargs):
        self.model_name = model_name
        self.max_retries = max_retries
        self.cost = 0.0
        self.kwargs = kwargs

    def generate(self, sys_prompt, query, image):
        """
        Send a single multimodal query and return the model's text response.

        Args:
            sys_prompt: System prompt string (can be empty).
            query:      The text query, possibly containing <ImageHere>.
            image:      A PIL Image object.

        Returns:
            The model's response text (str).
        """
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        content_parts = build_content_parts(query, image)
        messages.append({"role": "user", "content": content_parts})

        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.kwargs.get("temperature", 0.0),
        }

        litellm.success_callback = ["langfuse"] if LANGFUSE else []
        litellm.failure_callback = ["langfuse"] if LANGFUSE else []

        text = ""
        for attempt in range(self.max_retries):
            try:
                response = litellm.completion(**params)
                text = (response.choices[0].message.content or "").strip()
                try:
                    self.cost += float(completion_cost(completion_response=response))
                except Exception:
                    pass
                break
            except Exception as e:
                print(f"API error on attempt {attempt+1}/{self.max_retries}: {e}")
                time.sleep(min(2 ** attempt, 30))

        return text
