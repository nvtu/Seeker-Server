from time import strftime
from pydantic import BaseModel
from typing import List



class VideoInformationResponse(BaseModel): 
    shot_id: str
    keyframe_list: List[str] = []


class QueryResponseBaseModel(BaseModel):
    state_id: str
    query: str
    ranked_list: List[VideoInformationResponse] = []


class QueryByTextResponse(BaseModel):
    result: str
    reply: QueryResponseBaseModel