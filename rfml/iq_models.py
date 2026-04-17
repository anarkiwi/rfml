"""EfficientNet models adapted for 1-D IQ input (B, 2, N).

timm's EfficientNet expects 4-D input (B, C, H, W).  We add a singleton
height dimension inside forward so callers can pass the natural (B, 2, N)
shape that the rest of the pipeline produces.

State-dict keys are those of the underlying timm model, accessible via
model._model.  export_model.convert_model strips the "mdl." prefix added
by PyTorch-Lightning and then loads the resulting dict directly into this
class, whose own state_dict() is keyed as "_model.<timm_key>".
"""

import os
import torch
import torch.nn as nn
import timm


class _EfficientNetIQ(nn.Module):
    def __init__(
        self,
        variant: str,
        num_classes: int,
        drop_path_rate: float,
        drop_rate: float,
    ):
        super().__init__()
        self._model = timm.create_model(
            variant,
            pretrained=False,
            in_chans=2,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        )

    @property
    def classifier(self) -> nn.Linear:
        return self._model.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, N) — insert H=1 so Conv2d layers see (B, 2, 1, N)
        return self._model(x.unsqueeze(2))


def efficientnet_b0(
    num_classes: int = 53,
    pretrained: bool = False,
    path: str | None = None,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.2,
) -> _EfficientNetIQ:
    model = _EfficientNetIQ(
        "efficientnet_b0",
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
    )
    if pretrained and path and os.path.isfile(path):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model._model.load_state_dict(state_dict, strict=False)
    return model


def efficientnet_b4(
    num_classes: int = 53,
    pretrained: bool = False,
    path: str | None = None,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.2,
) -> _EfficientNetIQ:
    model = _EfficientNetIQ(
        "efficientnet_b4",
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
    )
    if pretrained and path and os.path.isfile(path):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model._model.load_state_dict(state_dict, strict=False)
    return model
