from fastapi import FastAPI
from models.request_models import QueryByText
from models.response_models import QueryByTextResponse
import numpy as np
from utils.search_utils import do_search



app = FastAPI(name='ECIR23 Server')


@app.post('/query', response_model = QueryByTextResponse, )
async def query_by_text(query_by_text: QueryByText):
    text_query = query_by_text.query
    user_id = query_by_text.user_id
    state_id = query_by_text.state_id

    reply = do_search(user_id, state_id, text_query)

    # Form response
    response = QueryByTextResponse(
        result = 'success',
        reply = reply
    )

    return response