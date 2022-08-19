from pydantic import BaseModel


class QueryByText(BaseModel):
    user_id: str
    state_id: str
    query: str

