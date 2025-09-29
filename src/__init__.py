# MOMbot package initialization
# This package provides audio transcription and speaker diarization capabilities

__all__ = [
    "audio_processing",
    "diarization",
    "evaluation",
    "main",
    "manifest",
    "openai_integration",
    "preprocessing",
    "utils"
]

# Lazy imports to avoid loading heavy dependencies at package import time
def __getattr__(name):
    if name == "audio_processing":
        from . import audio_processing
        return audio_processing
    elif name == "diarization":
        from . import diarization
        return diarization
    elif name == "evaluation":
        from . import evaluation
        return evaluation
    elif name == "main":
        from . import main
        return main
    elif name == "manifest":
        from . import manifest
        return manifest
    elif name == "openai_integration":
        from . import openai_integration
        return openai_integration
    elif name == "preprocessing":
        from . import preprocessing
        return preprocessing
    elif name == "utils":
        from . import utils
        return utils
    else:
        raise AttributeError(f"module 'src' has no attribute '{name}'")