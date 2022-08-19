import base64
import time


def generate_hash(salt: str = ""):
    """
    Generate a hash string.
    """
    current_time = str(time.time())
    key = base64.b64encode((current_time + '/' + salt).encode("ascii")).decode("ascii")
    return key


def generate_state_hash(user_id: str, query: str, engine = 'clip'):
    info = '.'.join([user_id, query, engine])
    state_id = generate_hash(info)
    return state_id