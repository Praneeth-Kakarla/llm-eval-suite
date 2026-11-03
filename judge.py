import asyncio
import time
from deepeval.models import DeepEvalBaseLLM
from google.genai import types
from my_bot import client, MODEL, _fallback_models


class GeminiJudge(DeepEvalBaseLLM):
    def load_model(self):
        return client

    def _call(self, prompt, schema=None):
        kwargs = {"http_options": types.HttpOptions(timeout=60000)}
        if schema is not None:
            kwargs.update(response_mime_type="application/json", response_schema=schema)
        config = types.GenerateContentConfig(**kwargs)
        models = [MODEL, "gemini-flash-lite-latest", "gemini-3.5-flash-lite"]
        last_err = None
        for attempt in range(6):
            model = models[attempt % len(models)]
            try:
                resp = client.models.generate_content(model=model, contents=prompt, config=config)
                return schema.model_validate_json(resp.text) if schema is not None else resp.text
            except Exception as e:
                last_err = e
                if "503" in str(e) or "429" in str(e):
                    time.sleep(2)
        raise RuntimeError(f"Judge failed: {last_err}")

    def generate(self, prompt, schema=None):
        return self._call(prompt, schema)

    async def a_generate(self, prompt, schema=None):
        return await asyncio.to_thread(self._call, prompt, schema)

    def get_model_name(self):
        return MODEL
