"""DenseHead: a per-timestep linear classifier (no pooling)."""

from torchdyno.heads.regression import RegressionHead
from torchdyno.registry import register_head


@register_head("dense")
class DenseHead(RegressionHead):
    """Per-timestep linear classifier: ``(T, B, H) -> (T, B, num_classes)``.

    The dense analogue of :class:`ClassificationHead` (which pools over time).
    Mechanically a per-timestep linear map, so it reuses :class:`RegressionHead`;
    the class exists to name the classification intent and register as
    ``"dense"``. Pair with :class:`~torchdyno.tasks.DenseLabeling`.
    """

    def __init__(self, input_size: int, num_classes: int, trainable: bool = True):
        super().__init__(input_size=input_size, output_size=num_classes, trainable=trainable)
