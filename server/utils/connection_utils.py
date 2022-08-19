import requests

def send(url, payload):
    response = requests.request(
        "POST", 
        url, 
        headers={'Content-Type': 'application/json'},
        data=payload
    )
    return response