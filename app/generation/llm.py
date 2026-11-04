from ollama import Client

from app.generation.prompts import (
    SYSTEM_PROMPT,
    build_rag_prompt
)

class OllamaLLM:

    def __init__(self):

        self.client = Client(
            host= "http://localhost:11434"
        )

        self.model = "llama3.2"

    def generate_answer(
            self,
            question:str,
            context:str
    ) -> str:

        prompt = build_rag_prompt(
            question= question,
            context=context
        )

        response = self.client.chat(
            model = self.model,
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