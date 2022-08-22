from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.request_models import GetAllFrames, QueryByText
from models.response_models import GetAllFramesResponse, QueryByTextResponse
import numpy as np
from utils.search_utils import do_search, get_all_frames_from_video



app = FastAPI(name='ECIR23 Server')

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/search', response_model = QueryByTextResponse)
async def query_by_text(request: QueryByText):
    text_query = request.query
    user_id = request.user_id
    state_id = request.state_id

    reply = do_search(user_id, state_id, text_query)

    # Form response
    response = QueryByTextResponse(
        result = 'success',
        reply = reply
    )

    return response


@app.post('/get_all_frames', response_model = GetAllFramesResponse)
async def get_all_frames(request: GetAllFrames):
    shot_id = request.shot_id
    all_frames_from_video = get_all_frames_from_video(shot_id)
    
    # Form response
    response = GetAllFramesResponse(
        result = 'success',
        reply = all_frames_from_video,
    )

    return response