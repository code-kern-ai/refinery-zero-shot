# -*- coding: utf-8 -*-

from fastapi import FastAPI, responses, status, Request
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
        "--- Running on CPU. If you're facing performance issues, you should consider switching to a CUDA device",
        flush=True,
    )


@app.middleware("http")
async def handle_db_session(request: Request, call_next):
    session_token = general.get_ctx_token()

    request.state.session_token = session_token
    try:
        response = await call_next(request)
    finally:
        general.remove_and_refresh_session(session_token)

    return response


@app.post("/zero-shot/text")
def zero_shot_text(
    request: request_classes.TextRequest,
) -> responses.JSONResponse:
    return_values = util.get_zero_shot_labels(
        request.project_id,
        request.config,
        request.label_names,
        request.text,
        request.run_individually,
        request.information_source_id,
    )
    return responses.JSONResponse(
        status_code=status.HTTP_200_OK,
        content=return_values,
    )


@app.post("/zero-shot/sample-records")
def zero_shot_text(
    request: request_classes.SampleRecordsRequest,
) -> responses.JSONResponse:
    return_values = util.get_zero_shot_10_records(
        request.project_id, request.information_source_id, request.label_names
    )
    return responses.JSONResponse(
        status_code=status.HTTP_200_OK,
        content=return_values,
    )


@app.post("/zero-shot/project")
def zero_shot_project(
    request: request_classes.ProjectRequest,
) -> responses.PlainTextResponse:
    # since this a "big" request session refresh logic in method itself
    util.zero_shot_project(request.project_id, request.payload_id)
    return responses.PlainTextResponse(status_code=status.HTTP_200_OK)


@app.get("/recommend")
def recommendations() -> responses.JSONResponse:
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

    return responses.JSONResponse(
        status_code=status.HTTP_200_OK,
        content=recommends,
    )


@app.put("/config_changed")
def config_changed() -> responses.PlainTextResponse:
    config_handler.refresh_config()
    return responses.PlainTextResponse(status_code=status.HTTP_200_OK)


@app.get("/healthcheck")
def healthcheck() -> responses.PlainTextResponse:
    text = ""
    status_code = status.HTTP_200_OK
    database_test = general.test_database_connection()
    if not database_test.get("success"):
        error_name = database_test.get("error")
        text += f"database_error:{error_name}:"
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    if not text:
        text = "OK"
    return responses.PlainTextResponse(text, status_code=status_code)
