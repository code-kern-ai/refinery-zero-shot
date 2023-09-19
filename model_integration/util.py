from enum import Enum


class ModelImplementation(Enum):
    GENERIC = "GENERIC"


__lookup_model_implementation = {}

__lookup_model_hypothesis = {"sahajtomar/german_zeroshot": "In diesem geht es um {}."}


def lookup_implementation_routine(config: str):
    config = config.lower()
    if config in __lookup_model_implementation:
        return __lookup_model_implementation[config]
    return ModelImplementation.GENERIC


def lookup_hypothesis(config: str):
    config = config.lower()
    if config in __lookup_model_hypothesis:
        return __lookup_model_hypothesis[config]
    return None
