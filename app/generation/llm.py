import logging
import os
from time import perf_counter

from ollama import Client

from app.generation.prompts import SYSTEM_PROMPT, build_rag_prompt

logger = logging.getLogger(__name__)


class OllamaLLM:

    def __init__(self):

        self.client = Client(
            host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )

        self.model = os.getenv("OLLAMA_MODEL", "llama3.2")

    def _chat(self, stage: str, **parameters):
        request = {"model": self.model, **parameters}
        logger.info("Ollama %s started: model=%s", stage, self.model)
        logger.debug("Ollama %s full request: %r", stage, request)
        started = perf_counter()
        try:
            response = self.client.chat(**request)
        except Exception:
            logger.exception("Ollama %s failed after %.3fs", stage, perf_counter() - started)
            raise
        logger.info("Ollama %s completed in %.3fs", stage, perf_counter() - started)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Ollama %s full response: %r", stage, response.model_dump())
        return response

    def generate_answer(
            self,
            question:str,
            context:str
    ) -> str:

        prompt = build_rag_prompt(
            question= question,
            context=context
        )

        response = self._chat(
            stage="answer",
            stream=False,
            messages=[
                {
                    "role": "system",
                    "content":SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options= {
                "temperature":0.1
            }
        )

        return response.message.content or ""     