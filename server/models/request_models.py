from pydantic import BaseModel


class QueryByText(BaseModel):
    user_id: str
    state_id: str
    query: str


class GetAllFrames(BaseModel):
    shot_id: str