"""Official AntiDeepfake inference (model-card code), run in the isolated fairseq environment.

Usage: data/tts-envs/fairseq/bin/python scripts/reference/antideepfake_fairseq.py MODEL_DIR OUT.json clip...
The model classes and preprocessing are copied from the model card at the pinned revision; weights load
from the local, checksum-verified model.safetensors instead of a network download. CPU only.
"""
import json
import sys

import torch
import torchaudio
from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config
from safetensors.torch import load_file

device = torch.device('cpu')


class SSLModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        cfg = Wav2Vec2Config(
            quantize_targets=True, extractor_mode="layer_norm", layer_norm_first=True, final_dim=768,
            latent_temp=(2.0, 0.1, 0.999995), encoder_layerdrop=0.0, dropout_input=0.0, dropout_features=0.0,
            dropout=0.0, attention_dropout=0.0, conv_bias=True, encoder_layers=24, encoder_embed_dim=1024,
            encoder_ffn_embed_dim=4096, encoder_attention_heads=16, feature_grad_mult=1.0)
        self.model = Wav2Vec2Model(cfg)

    def extract_feat(self, input_data):
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]
        with torch.no_grad():
            features = self.model(input_data.to(device), mask=False, features_only=True)['x']
        return features


def load_wav_and_preprocess(wav_path, target_sr=16000):
    wav, sr = torchaudio.load(wav_path)
    wav = wav.mean(dim=0)
    wav = torchaudio.functional.resample(wav, sr, new_freq=target_sr)
    with torch.no_grad():
        wav = torch.nn.functional.layer_norm(wav, wav.shape)
    return wav.unsqueeze(0).to(device)


class DeepfakeDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ssl_orig_output_dim = 1024
        self.num_classes = 2
        self.m_ssl = SSLModel()
        self.adap_pool1d = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.proj_fc = torch.nn.Linear(in_features=self.ssl_orig_output_dim, out_features=self.num_classes)

    def forward(self, wav):
        emb = self.m_ssl.extract_feat(wav)
        emb = emb.transpose(1, 2)
        pooled_emb = self.adap_pool1d(emb).squeeze(-1)
        return self.proj_fc(pooled_emb)


def main():
    model_dir, out, clips = sys.argv[1], sys.argv[2], sys.argv[3:]
    model = DeepfakeDetector()
    model.load_state_dict(load_file(f'{model_dir}/model.safetensors'), strict=True)
    model.eval()
    results = {}
    with torch.no_grad():
        for clip in clips:
            logits = model(load_wav_and_preprocess(clip))[0]
            results[clip] = {'logits_fake_real': [float(v) for v in logits],
                             'fake_prob': float(torch.softmax(logits, dim=0)[0])}
    with open(out, 'w') as stream:
        json.dump(results, stream, indent=2)


if __name__ == '__main__':
    main()
