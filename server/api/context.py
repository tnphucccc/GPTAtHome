from pydantic import BaseModel

from core.src.generate import RuntimeModel

model = RuntimeModel()


class Context(BaseModel):
    prompt: str
    max_tokens: int

    def response(self) -> dict:
        story = model.request(self.prompt)
        return {"response": story}
