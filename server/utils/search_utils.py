import json
import numpy as np
import os
from utils.hash_utils import generate_state_hash
from utils.connection_utils import send
from utils.response_utils import process_milvus_search_results, cache_query_result
from server_managers import collection, milvus_search_params, CLIP_SERVER_URL, all_frames_mapping
from models.response_models import QueryResponseBaseModel
from typing import List


def encode_text(text_query):
    payload = json.dumps({ 'text_query': text_query })
    response = send(CLIP_SERVER_URL, payload)
    return response.json()


def search_by_embedded_text(embedded_text):

    LIMIT = 1000 # The maximum number of results to return

    results = collection.search(
        data = [embedded_text],
        anns_field = 'embedding',
        param = milvus_search_params,
        limit = LIMIT,
        expr = None,
        consistent_level = "Strong",
    )

    return results


def search_by_text(text_query):
    # Encode text
    encoded_text = encode_text(text_query)
    embedded_text = np.array(encoded_text['text_embedding'])

    # Search by milvus
    results = search_by_embedded_text(embedded_text)
    ranked_list = process_milvus_search_results(results)

    return ranked_list


def do_search(user_id: str, state_id: str, text_query) -> QueryResponseBaseModel:
    ranked_list = search_by_text(text_query)
    state_id = generate_state_hash(user_id, text_query)
    cache_query_result(state_id, text_query, ranked_list)
    
    response = QueryResponseBaseModel(
        state_id = state_id,
        query = text_query,
        ranked_list = ranked_list
    )

    return response


def get_all_frames_from_video(shot_id: str) -> List[str]:
    frames = [os.path.splitext(frame)[0] for frame in all_frames_mapping[shot_id]]
    return frames