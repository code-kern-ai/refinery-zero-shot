import os
import requests


MODEL_PROVIDER_BASE_URI = os.getenv("MODEL_PROVIDER")


def get_model_path(model_name: str, revision: str = None) -> str:
    url = f"{MODEL_PROVIDER_BASE_URI}/model_path"
    params = {
        "model_name": model_name,
        "revision": revision,
    }
    response = requests.get(url, params=params)
    return response.json()
