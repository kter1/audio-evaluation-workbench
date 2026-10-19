"""
Pellav2 audio deepfake detector
https://huggingface.co/Sadanie/pellav2-audio-deepfake-detector
Revision a4c68a3d4afc7f5d9450c94f5d0652aaa208d40a, pellav2_infer.py
MIT license (model card metadata); see PELLAV2-LICENSE

Local change: the XLS-R backbone is constructed from its pinned configuration
instead of Wav2Vec2Model.from_pretrained. Every backbone weight comes from the
pellav2 checkpoint, so no pretrained backbone download is needed. Attribute
names are unchanged so checkpoint keys match.
"""
import torch
import torch.nn as nn


class Detector(nn.Module):
    def __init__(self, backbone_config):
        super().__init__()
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        self.backbone = Wav2Vec2Model(Wav2Vec2Config.from_dict(backbone_config))
        self.fusion = True
        self.layer_weights = nn.Parameter(
            torch.zeros(self.backbone.config.num_hidden_layers + 1))
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, x):
        hs = self.backbone(x, output_hidden_states=True).hidden_states
        w = torch.softmax(self.layer_weights, dim=0)
        h = torch.stack(hs).mul(w[:, None, None, None]).sum(0).mean(dim=1)
        return self.head(h).squeeze(-1)
