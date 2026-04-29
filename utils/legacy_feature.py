import numpy as np


LEGACY_INPUT_CHANNELS = 1
LEGACY_NUM_MELS = 64
DEFAULT_MAX_FRAMES = 300
LEGACY_FEATURE_SHAPE = (
    LEGACY_INPUT_CHANNELS,
    LEGACY_NUM_MELS,
    DEFAULT_MAX_FRAMES,
)
STATIC_LEGACY_INPUT_SHAPE = (1, *LEGACY_FEATURE_SHAPE)
STATIC_LEGACY_INPUT_SHAPE_TEXT = ",".join(str(item) for item in STATIC_LEGACY_INPUT_SHAPE)


def legacy_feature_shape_text(batch_symbol="N"):
    return (
        f"[{batch_symbol},{LEGACY_INPUT_CHANNELS},"
        f"{LEGACY_NUM_MELS},{DEFAULT_MAX_FRAMES}]"
    )


def normalize_legacy_feature_array(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4 or tuple(array.shape[1:]) != LEGACY_FEATURE_SHAPE:
        raise ValueError(
            f"Expected input {legacy_feature_shape_text()}, got {tuple(array.shape)}"
        )
    return np.ascontiguousarray(array)
