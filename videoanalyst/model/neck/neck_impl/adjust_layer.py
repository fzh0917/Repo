import torch
import torch.nn as nn

from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.neck.neck_base import TRACK_NECKS


@TRACK_NECKS.register
class AdjustLayer(ModuleBase):
    default_hyper_params = dict(
        in_channels=768,
        out_channels=512,
    )

    def __init__(self):
        super(AdjustLayer, self).__init__()

    def forward(self, x):
        return self.adjustor(x)

    def update_params(self):
        super().update_params()
        in_channels = self._hyper_params['in_channels']
        out_channels = self._hyper_params['out_channels']
        self.adjustor = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self._init_weights()

    def _init_weights(self):
        conv_weight_std = 0.01
        for m in [self.adjustor, ]:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=conv_weight_std)  # conv_weight_std=0.01

