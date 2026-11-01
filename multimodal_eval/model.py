"""
Multimodal model interface for ChemEval.

Supports vision-capable LLMs via litellm (GPT-4o, Claude, Gemini, etc.), local
open-weight vision models via vLLM, and the OpenAI / Anthropic batch APIs.
Images are sent inline as base64-encoded data URIs alongside the text query.
"""
import os
import io
import re
import json
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


class HuggingFaceMultimodalModel:
    """Open-weight vision LLM served locally with vLLM (e.g. Qwen2-VL, InternVL, LLaVA)."""

    def __init__(self, temperature, model_name, max_new_tokens=2048, max_model_len=None):
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.temperature = temperature
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        kw = {"max_model_len": max_model_len} if max_model_len else {}
        self.llm = LLM(model=model_name, trust_remote_code=True,
                       limit_mm_per_prompt={"image": 1}, **kw)

    def build_messages(self, sys_prompt, query, image):
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": build_content_parts(query, image)})
        return messages

    def generate_batch(self, sys_prompt, queries, images) -> list[str]:
        conversations = [self.build_messages(sys_prompt, q, im) for q, im in zip(queries, images)]
        params = self.SamplingParams(
            temperature=self.temperature, top_p=1.0, max_tokens=self.max_new_tokens)
        # llm.chat applies the model's chat template and loads the images itself
        results = self.llm.chat(conversations, params, use_tqdm=False)   # same order as input
        return [r.outputs[0].text.strip() for r in results]

    def generate(self, sys_prompt, query, image) -> str:
        return self.generate_batch(sys_prompt, [query], [image])[0]


# ---------------------------------------------------------------------------
# Batch APIs (Anthropic Message Batches / OpenAI Batch). No temperature is sent.
# ---------------------------------------------------------------------------
class MultimodalBatchModel:
    """Submit all (query, image) pairs as ONE asynchronous batch job and fetch it later.

    submit()  -> batch id
    status()  -> (state, info)  state is "running", "done" or "failed"
    results() -> (answers, errors)  answers: {custom_id: text}, errors: {custom_id: msg}
    """
    MAX_REQUESTS = 0
    MAX_BYTES = 0

    def _build(self, sys_prompt, query, image, custom_id):
        raise NotImplementedError

    def _send(self, requests) -> str:
        raise NotImplementedError

    def submit(self, sys_prompt, items) -> str:
        """items: iterable of (custom_id, query, PIL image)."""
        requests, size = [], 0
        for cid, query, image in items:
            req = self._build(sys_prompt, query, image, cid)
            requests.append(req)
            size += len(json.dumps(req, ensure_ascii=False).encode("utf-8")) + 1
        if len(requests) > self.MAX_REQUESTS:
            raise ValueError(f"{len(requests)} requests exceeds the batch limit of {self.MAX_REQUESTS}")
        if size > self.MAX_BYTES:
            raise ValueError(f"Batch is {size / 1e6:.0f} MB, over the ~{self.MAX_BYTES / 1e6:.0f} MB "
                             "limit. Use --limit or downscale/compress the images.")
        return self._send(requests)

    def status(self, batch_id):
        raise NotImplementedError

    def results(self, batch_id):
        raise NotImplementedError


def _to_anthropic_parts(parts):
    """OpenAI-style content parts -> Anthropic content blocks."""
    out = []
    for p in parts:
        if p["type"] == "text":
            out.append({"type": "text", "text": p["text"]})
        else:
            header, b64 = p["image_url"]["url"].split(",", 1)      # data:image/png;base64
            media_type = header[len("data:"):].split(";")[0]
            out.append({"type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64}})
    return out


class AnthropicMultimodalBatchModel(MultimodalBatchModel):
    MAX_REQUESTS = 100_000
    MAX_BYTES = 256_000_000          # API limit is 256 MB per batch

    def __init__(self, model_name, max_new_tokens=2048):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens or 2048   # max_tokens is mandatory for Anthropic

    def _build(self, sys_prompt, query, image, custom_id):
        params = {
            "model": self.model_name,
            "max_tokens": self.max_new_tokens,
            "messages": [{"role": "user",
                          "content": _to_anthropic_parts(build_content_parts(query, image))}],
        }
        if sys_prompt:
            params["system"] = sys_prompt
        return {"custom_id": custom_id, "params": params}

    def _send(self, requests):
        batch = self.client.messages.batches.create(requests=requests)
        print(f"Anthropic batch created with {len(requests)} requests. Batch ID: {batch.id}")
        return batch.id

    def status(self, batch_id):
        b = self.client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        info = (f"{b.processing_status}: processing={c.processing} succeeded={c.succeeded} "
                f"errored={c.errored} canceled={c.canceled} expired={c.expired}")
        return ("done" if b.processing_status == "ended" else "running"), info

    def results(self, batch_id):
        answers, errors = {}, {}
        for entry in self.client.messages.batches.results(batch_id):
            res = entry.result
            if res.type == "succeeded":
                text = "".join(b.text for b in res.message.content
                               if getattr(b, "type", None) == "text")
                answers[entry.custom_id] = text.strip()
            elif res.type == "errored":
                errors[entry.custom_id] = str(getattr(res, "error", "errored"))
            else:                                    # canceled / expired
                errors[entry.custom_id] = res.type
        return answers, errors


class OpenAIMultimodalBatchModel(MultimodalBatchModel):
    MAX_REQUESTS = 50_000
    MAX_BYTES = 200_000_000          # API limit is 200 MB per input file

    def __init__(self, model_name, max_new_tokens=None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model_name = model_name
        # Optional. Reasoning models count thinking tokens against this limit,
        # so leave it as None unless you really want a cap.
        self.max_new_tokens = max_new_tokens

    def _build(self, sys_prompt, query, image, custom_id):
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": build_content_parts(query, image)})
        body = {"model": self.model_name, "messages": messages}
        if self.max_new_tokens:
            body["max_completion_tokens"] = self.max_new_tokens
        return {"custom_id": custom_id, "method": "POST",
                "url": "/v1/chat/completions", "body": body}

    def _send(self, requests):
        data = ("\n".join(json.dumps(r, ensure_ascii=False) for r in requests) + "\n").encode("utf-8")
        up = self.client.files.create(file=("batch_input.jsonl", data), purpose="batch")
        batch = self.client.batches.create(input_file_id=up.id,
                                           endpoint="/v1/chat/completions",
                                           completion_window="24h")
        print(f"OpenAI batch created with {len(requests)} requests. Batch ID: {batch.id}")
        return batch.id

    def status(self, batch_id):
        b = self.client.batches.retrieve(batch_id)
        rc = b.request_counts
        info = f"{b.status}"
        if rc:
            info += f": completed={rc.completed} failed={rc.failed} total={rc.total}"
        if b.status in ("completed", "expired", "cancelled"):
            return "done", info          # expired/cancelled may still hold partial output
        if b.status == "failed":
            return "failed", info + f" errors={getattr(b, 'errors', None)}"
        return "running", info

    def _read_file(self, file_id):
        if not file_id:
            return []
        text = self.client.files.content(file_id).text
        return [json.loads(l) for l in text.splitlines() if l.strip()]

    def results(self, batch_id):
        b = self.client.batches.retrieve(batch_id)
        answers, errors = {}, {}
        for rec in self._read_file(b.output_file_id):
            cid = rec["custom_id"]
            resp = rec.get("response") or {}
            if rec.get("error") or resp.get("status_code") != 200:
                errors[cid] = str(rec.get("error") or resp.get("body"))
                continue
            content = resp["body"]["choices"][0]["message"].get("content") or ""
            answers[cid] = content.strip()
        for rec in self._read_file(b.error_file_id):
            errors.setdefault(rec["custom_id"], str(rec.get("error") or rec.get("response")))
        return answers, errors