import os
import time
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
        if self.max_new_tokens:
            params["max_tokens"] = self.max_new_tokens
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