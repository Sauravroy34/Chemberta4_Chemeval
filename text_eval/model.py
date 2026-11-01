import os
import time
from dotenv import load_dotenv
import litellm
from litellm import completion_cost
import json
load_dotenv(override=True)
litellm.drop_params = True

LANGFUSE = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    from langfuse import get_client
    LANGFUSE = get_client()
else:
    print("WARNING: Langfuse keys not set - nothing will be logged to Langfuse")


class ModelInterface:
    def generate(self, sys_prompt: str, user_prompt: str, n: int = 1) -> list[str]:
        raise NotImplementedError


class LiteLLMModel(ModelInterface):
    def __init__(self, model_name, max_new_tokens=None, max_retries=6, **kwargs):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.cost = 0.0
        self.kwargs = kwargs

    def _params(self, sys_prompt, user_prompt):
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": user_prompt})
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.kwargs.get("temperature", 0.0),
        }
        # if self.max_new_tokens:
        #     params["max_tokens"] = self.max_new_tokens
        return params

    def generate(self, sys_prompt, user_prompt, n=1) -> list[str]:
        params = self._params(sys_prompt, user_prompt)
        litellm.success_callback = ["langfuse"] if LANGFUSE else []
        litellm.failure_callback = ["langfuse"] if LANGFUSE else []
        outputs = []
        for _ in range(n):
            text = ""
            for attempt in range(self.max_retries):
                try:
                    response = litellm.completion(**params)
                    text = (response.choices[0].message.content or "").strip()
                    try:
                        self.cost += float(completion_cost(completion_response=response))
                    except Exception:
                        pass  # cost tracking must never discard a good answer
                    break
                except Exception as e:
                    print(f"API error on attempt {attempt+1}/{self.max_retries}: {e}")
                    time.sleep(min(2 ** attempt, 30))
            outputs.append(text)
        return outputs


class HuggingFaceModel(ModelInterface):
    def __init__(self, temperature, model_name, max_new_tokens=2048, max_model_len=None):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.temperature = temperature
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        kw = {"max_model_len": max_model_len} if max_model_len else {}
        self.llm = LLM(model=model_name, trust_remote_code=True, **kw)

    def build_prompt(self, sys_prompt, user_prompt):
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": user_prompt})
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:  # model without a chat template
            print(f"No chat template, using raw prompt: {e}")
            return (sys_prompt + "\n" if sys_prompt else "") + user_prompt

    def generate_batch(self, sys_prompt, user_prompts) -> list[str]:
        prompts = [self.build_prompt(sys_prompt, u) for u in user_prompts]
        params = self.SamplingParams(
            temperature=self.temperature, top_p=1.0, max_tokens=self.max_new_tokens)
        results = self.llm.generate(prompts, params)       # same order as input
        return [r.outputs[0].text.strip() for r in results]

    def generate(self, sys_prompt, user_prompt, n=1) -> list[str]:
        return [self.generate_batch(sys_prompt, [user_prompt])[0] for _ in range(n)]


# ---------------------------------------------------------------------------
# Batch APIs (Anthropic Message Batches / OpenAI Batch). No temperature is sent.
# ---------------------------------------------------------------------------
class BatchModelInterface:
    """Submit many prompts as one asynchronous batch job and fetch them later.

    status()  -> (state, info)  state is "running", "done" or "failed"
    results() -> (answers, errors)  answers: {custom_id: text}, errors: {custom_id: msg}
    """

    def submit(self, sys_prompt: str, queries: list[str], custom_ids: list[str]) -> str:
        raise NotImplementedError

    def status(self, batch_id: str):
        raise NotImplementedError

    def results(self, batch_id: str):
        raise NotImplementedError


class AnthropicBatchModel(BatchModelInterface):
    MAX_REQUESTS = 100_000

    def __init__(self, model_name, max_new_tokens=32000):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens or 32000   # max_tokens is mandatory for Anthropic

    def submit(self, sys_prompt, queries, custom_ids):
        if len(queries) > self.MAX_REQUESTS:
            raise ValueError(f"Anthropic batches allow at most {self.MAX_REQUESTS} requests")
        requests = []
        for cid, q in zip(custom_ids, queries):
            params = {
                "model": self.model_name,
                "max_tokens": self.max_new_tokens,
                "messages": [{"role": "user", "content": q}],
            }
            if sys_prompt:
                params["system"] = sys_prompt
            requests.append({"custom_id": cid, "params": params})
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


class OpenAIBatchModel(BatchModelInterface):
    MAX_REQUESTS = 50_000

    def __init__(self, model_name, max_new_tokens=None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model_name = model_name
        # Optional. Reasoning models count thinking tokens against this limit,
        # so leave it as None unless you really want a cap.
        self.max_new_tokens = max_new_tokens

    def submit(self, sys_prompt, queries, custom_ids):
        if len(queries) > self.MAX_REQUESTS:
            raise ValueError(f"OpenAI batches allow at most {self.MAX_REQUESTS} requests")
        lines = []
        for cid, q in zip(custom_ids, queries):
            messages = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": q})
            body = {"model": self.model_name, "messages": messages}
            if self.max_new_tokens:
                body["max_completion_tokens"] = self.max_new_tokens
            lines.append(json.dumps({"custom_id": cid, "method": "POST",
                                     "url": "/v1/chat/completions", "body": body},
                                    ensure_ascii=False))
        data = ("\n".join(lines) + "\n").encode("utf-8")
        up = self.client.files.create(file=("batch_input.jsonl", data), purpose="batch")
        batch = self.client.batches.create(input_file_id=up.id,
                                           endpoint="/v1/chat/completions",
                                           completion_window="24h")
        print(f"OpenAI batch created with {len(lines)} requests. Batch ID: {batch.id}")
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