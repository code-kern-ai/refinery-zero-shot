# -*- coding: utf-8 -*-
from fastapi import FastAPI
from typing import List, Dict, Tuple
import torch
import request_classes
from submodules.model.business_objects import general
from util import util, config_handler

app = FastAPI()

if torch.cuda.is_available():
    print(
        f"--- Running with GPU acceleration: {torch.cuda.get_device_name(torch.cuda.current_device())}",
        flush=True,
    )
else:
    print(
        f"--- Running on CPU. If you're facing performance issues, you should consider switching to a CUDA device",
        flush=True,
    )


@app.post("/zero-shot/text")
def zero_shot_text(
    request: request_classes.TextRequest,
) -> Tuple[List[Tuple[str, float]], int]:
    session_token = general.get_ctx_token()
    return_values = util.get_zero_shot_labels(
        request.project_id,
        request.config,
        request.label_names,
        request.text,
        request.run_individually,
        request.information_source_id,
    )
    general.remove_and_refresh_session(session_token)
    return return_values, 200


@app.post("/zero-shot/sample-records")
def zero_shot_text(
    request: request_classes.SampleRecordsRequest,
) -> Tuple[List[Tuple[str, float]], int]:
    session_token = general.get_ctx_token()
    return_values = util.get_zero_shot_10_records(
        request.project_id, request.information_source_id, request.label_names
    )
    general.remove_and_refresh_session(session_token)
    return return_values, 200


@app.post("/zero-shot/project")
def zero_shot_project(request: request_classes.ProjectRequest) -> Tuple[str, int]:
    # since this a "big" request session refesh logic in method itself
    util.zero_shot_project(request.project_id, request.payload_id)
    return "", 200


@app.get("/recommend")
def recommendations() -> Tuple[List[Dict[str, str]], int]:
    recommends = [
        {
            "configString": "Sahajtomar/German_Zeroshot",
            "avgTime": "~ 25 char per sec",
            "language": "de",
            "link": "https://huggingface.co/Sahajtomar/German_Zeroshot",
            "base": "GBERT Large",
            "size": "1.25 GB",
            "prio": 1,
        },
        {
            "configString": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            "avgTime": "~ 20 char per sec",
            "language": "de",
            "link": "https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            "base": "CC100 multilingual",
            "size": "1.04 GB",
            "prio": 2,
        },
        {
            "configString": "Narsil/deberta-large-mnli-zero-cls",
            "avgTime": "~ 25 char per sec",
            "language": "en",
            "link": "https://huggingface.co/Narsil/deberta-large-mnli-zero-cls",
            "base": "DeBERTa",
            "size": "1.51 GB",
            "prio": 3,
        },
        {
            "configString": "typeform/distilbert-base-uncased-mnli",
            "avgTime": "~ 275 char per sec",
            "language": "en",
            "link": "https://huggingface.co/typeform/distilbert-base-uncased-mnli",
            "base": "uncased DistilBERT",
            "size": "255 MB",
            "prio": 1,
        },
        {
            "configString": "cross-encoder/nli-distilroberta-base",
            "avgTime": "~ 230 char per sec",
            "language": "en",
            "link": "https://huggingface.co/cross-encoder/nli-distilroberta-base",
            "base": "SNLI and MultiNLI",
            "size": "313 MB",
            "prio": 2,
        },
    ]

    return recommends, 200


@app.put("/config_changed")
def config_changed() -> int:
    config_handler.refresh_config()
    return 200
