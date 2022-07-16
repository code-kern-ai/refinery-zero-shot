from .util import ModelImplementation, lookup_implementation_routine
from .models import generic

from typing import List, Dict, Any, Optional


def get_labels_for_text(
    project_id: str,
    config: str,
    text: str,
    labels: List[str],
    information_source_id: Optional[str] = None,
) -> Dict[str, Any]:
    target = lookup_implementation_routine(config)
    if target == ModelImplementation.GENERIC:
        return generic.get_labels_for_text(
            project_id, config, text, labels, information_source_id
        )
    else:
        pass
    return None
