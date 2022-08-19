import os
import json
from collections import defaultdict
from fastapi.encoders import jsonable_encoder
from models.response_models import VideoInformationResponse
from typing import List
from server_managers import rd


def process_milvus_search_results(results) -> List[VideoInformationResponse]:
    images = list(results[0].ids)
    scores = list(results[0].distances)
    group_by_video = defaultdict(list)

    # Group ranked list frames by video
    for i, image_name in enumerate(images):
        video_name = image_name.split('-')[0]
        image_name = os.path.splitext(image_name)[0]
        group_by_video[video_name].append((image_name, scores[i]))
    
    # Sort frames by score
    videos = []
    for video_name, items in group_by_video.items():
        video_score = sum([score for (_, score) in items])
        ranked_frame_list = sorted(items, key=lambda x: x[1])
        videos.append((video_score, 
            VideoInformationResponse(
                shot_id = video_name,
                keyframe_list = [frame for (frame, _) in ranked_frame_list]
            )))
    
    # Sort videos by video score
    ranked_list = [video_information_response for (_, video_information_response) in sorted(videos, key=lambda x: x[0])]
    return ranked_list


def cache_query_result(state_id, query, ranked_list):
    rd.set(state_id, json.dumps({
        'query': query, 
        'ranked_list': jsonable_encoder(ranked_list)
    }))