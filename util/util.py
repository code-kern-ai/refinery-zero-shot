from submodules.model import enums
from submodules.model.business_objects import (
    general,
    information_source,
    project,
    record,
    record_label_association,
)
from typing import List, Optional
from model_integration.controller import get_labels_for_text
import traceback
from util.notification import send_project_update


def zero_shot_project(project_id: str, payload_id: str):
    try:
        config = project.get_zero_shot_project_config(project_id, payload_id)
        if not config:
            raise ValueError(
                f"Can't find config for project {project_id} & payload {payload_id}"
            )

        if len(config.label_names) == 0:
            raise ValueError(
                f"No labels found for project {project_id} & payload {payload_id}"
            )
        session_token = general.get_ctx_token()

        record_label_association.delete_by_source_id(
            project_id, config.source_id, with_commit=True
        )
        record_batches = record.get_record_id_groups(project_id)
        max_count = len(record_batches)
        label_dict = {lx[0]: lx[1] for lx in zip(config.label_names, config.label_ids)}
        count = 0
        is_cancelled = False
        for batch in record_batches:
            if count % 10 == 0:
                session_token = general.remove_and_refresh_session(session_token, True)
            progress = count / max_count
            information_source.update_payload(
                project_id, payload_id, progress=progress, with_commit=True
            )
            send_project_update(
                project_id, f"zero-shot:{payload_id}:progress:{progress}"
            )
            text_data = record.get_record_data_for_id_group(
                project_id, batch, config.attribute_name
            )
            for key in text_data:
                result = get_zero_shot_labels(
                    project_id,
                    config.config,
                    config.label_names,
                    text_data[key],
                    config.run_individually,
                )
                if result[0][1] > config.min_confidence:
                    record_label_association.create(
                        project_id=project_id,
                        record_id=key,
                        source_id=config.source_id,
                        source_type="INFORMATION_SOURCE",
                        return_type=enums.InformationSourceReturnType.RETURN.value,
                        confidence=result[0][1],
                        created_by=config.created_by,
                        labeling_task_label_id=label_dict[result[0][0]],
                        is_gold_star=False,
                        with_commit=None,
                    )

            if information_source.continue_payload(
                project_id, config.source_id, payload_id
            ):
                general.commit()
                count += 1
            else:
                is_cancelled = True
                record_label_association.delete_by_source_id(
                    project_id, config.source_id, True
                )
                session_token = general.remove_and_refresh_session(session_token, True)
                break

        state = (
            enums.PayloadState.FAILED.value
            if is_cancelled
            else enums.PayloadState.FINISHED.value
        )
        information_source.update_payload(
            project_id,
            payload_id,
            progress=1,
            state=state,
            with_commit=True,
        )
        send_project_update(
            project_id,
            f"zero-shot:{payload_id}:state:{state}",
        )
    except Exception:
        print(traceback.format_exc(), flush=True)
        session_token = general.remove_and_refresh_session(session_token, True)
        information_source.update_payload(
            project_id,
            payload_id,
            state=enums.PayloadState.FAILED.value,
            with_commit=True,
        )
        send_project_update(
            project_id,
            f"zero-shot:{payload_id}:state:{enums.PayloadState.FAILED.value}",
        )
    if session_token:
        general.remove_and_refresh_session(session_token, False)


def get_zero_shot_labels(
    project_id: str,
    config: str,
    labels: List[str],
    text: str,
    run_individually: bool,
    information_source_id: Optional[str] = None,
):
    return_values = []
    if run_individually:
        for label in labels:
            result = get_labels_for_text(
                project_id, config, text, [label], information_source_id
            )
            return_values.append((label, result["scores"][0]))
        return_values = sorted(return_values, key=lambda tup: tup[1], reverse=True)
    else:
        result = get_labels_for_text(
            project_id, config, text, labels, information_source_id
        )
        for label, confidence in zip(result["labels"], result["scores"]):
            return_values.append((label, confidence))
    return return_values


def get_zero_shot_10_records(
    project_id: str, information_source_id: str, label_names: Optional[List[str]] = None
):
    zero_shot_is = information_source.get(project_id, information_source_id)
    if not zero_shot_is:
        raise ValueError("unknown information source:" + information_source_id)
    result_set = information_source.get_zero_shot_is_data(
        project_id, information_source_id
    )
    if result_set.is_type != enums.InformationSourceType.ZERO_SHOT.value:
        raise ValueError("unknown information source type:" + result_set.is_type)
    record_set = record.get_zero_shot_n_random_records(
        project_id, result_set.attribute_name
    )
    if not label_names:
        label_names = result_set.labels
    result_records = []
    for record_item in record_set:
        result = get_zero_shot_labels(
            project_id,
            result_set.config,
            label_names,
            record_item.text,
            result_set.run_individually,
            information_source_id,
        )
        result_records.append(
            {
                "record_id": str(record_item.id),
                "labels": [{"label_name": x[0], "confidence": x[1]} for x in result],
                "checked_text": record_item.text,
                "full_record_data": record_item.data,
            }
        )
    return result_records
