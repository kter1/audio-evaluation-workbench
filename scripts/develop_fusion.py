"""Protocol v4 development: per-window features from both detectors on the already-seen v3 clips.

  render  write MP3, telephone, and 20 dB noise versions of all 360 v3 clips under data/v4-dev/channels
  score   record pellav2 and AntiDeepfake logits for every 4 s window of every clip version
          (resumable; data/v4-dev/features.json)
  noise   add each window's noise index (lab.fusion.noise_index) from the same audio; no model inference

The v3 clips were all scored and reported before v4 began, so they are development data here.
Nothing in this script touches the fresh v4 test set.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.store import digest
from scripts import build_modern_set as m

DEV = ROOT / 'data/v4-dev'
FEATURES = DEV / 'features.json'
V3_MANIFEST = ROOT / 'data/evaluation/modern-manifest.json'
CHANNELS = ('clean', 'mp3', 'phone', 'noise20')


def v3_cases():
    manifest = json.loads(V3_MANIFEST.read_text())
    if digest({k: v for k, v in manifest.items() if k != 'sha256'}) != manifest['sha256']:
        raise ValueError('v3 manifest changed')
    return manifest, manifest['cases']


def version_path(case, channel):
    if channel == 'clean':
        return ROOT / case['path']
    m.CHANNELS = ('mp3', 'phone', 'noise20')
    return m.channel_paths(case, work=DEV)[channel]


def stage_render():
    _, cases = v3_cases()
    (DEV / 'channels').mkdir(parents=True, exist_ok=True)
    for i, case in enumerate(cases, 1):
        for channel in CHANNELS[1:]:
            target = version_path(case, channel)
            if not target.exists():
                m.render(ROOT / case['path'], target, channel, f'v4-dev:{case["id"]}')
        if i % 60 == 0:
            print(f'render {i}/{len(cases)}', flush=True)
    print('Development channel versions ready.')


def stage_score():
    from lab.antideepfake import AntiDeepfakeDetector
    from lab.forensics import anchored_spans, mono16, read_audio
    from lab.pella import PellaV2Detector
    manifest, cases = v3_cases()
    pella, adf = PellaV2Detector(), AntiDeepfakeDetector()
    store = json.loads(FEATURES.read_text()) if FEATURES.exists() else {
        'v3_manifest_sha256': manifest['sha256'], 'models': {'pellav2': pella.manifest, 'antideepfake': adf.manifest},
        'channels': list(CHANNELS), 'rows': {}}
    if store['models'] != {'pellav2': pella.manifest, 'antideepfake': adf.manifest}:
        raise ValueError('Detector files changed since features were started')
    todo = [(c, ch) for c in cases for ch in CHANNELS if f'{c["id"]}|{ch}' not in store['rows']]
    for i, (case, channel) in enumerate(todo, 1):
        blob = version_path(case, channel).read_bytes()
        audio, rate, _ = read_audio(blob)
        x = mono16(audio, rate)
        windows = []
        for a, b in anchored_spans(len(x), pella.window_samples):
            native = adf.logits(x[a:b])
            windows.append({'start': a, 'end': b, 'pellav2_logit': pella.score(x[a:b])['logits'][1],
                            'antideepfake_logit': float(native[0] - native[1])})
        store['rows'][f'{case["id"]}|{channel}'] = {
            'id': case['id'], 'channel': channel, 'label': case['label'], 'generator': case['generator'],
            'source': case['source'], 'speaker': case['speaker'], 'sha256': hashlib.sha256(blob).hexdigest(),
            'windows': windows}
        if i % 40 == 0 or i == len(todo):
            FEATURES.write_text(json.dumps(store) + '\n')
            print(f'score {i}/{len(todo)}', flush=True)
    print(f'Features ready: {len(store["rows"])} clip versions.')


def stage_noise():
    from lab.forensics import mono16, read_audio
    from lab.fusion import noise_index
    _, cases = v3_cases()
    store = json.loads(FEATURES.read_text())
    by_id = {c['id']: c for c in cases}
    for i, (key, row) in enumerate(store['rows'].items(), 1):
        blob = version_path(by_id[row['id']], row['channel']).read_bytes()
        if hashlib.sha256(blob).hexdigest() != row['sha256']:
            raise ValueError(f'Audio changed since scoring: {key}')
        audio, rate, _ = read_audio(blob)
        x = mono16(audio, rate)
        for w in row['windows']:
            w['noise_index'] = noise_index(x[w['start']:w['end']])
        if i % 240 == 0:
            print(f'noise {i}/{len(store["rows"])}', flush=True)
    FEATURES.write_text(json.dumps(store) + '\n')
    print('Noise index added to every development window.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['render', 'score', 'noise'])
    {'render': stage_render, 'score': stage_score, 'noise': stage_noise}[parser.parse_args().stage]()


if __name__ == '__main__':
    main()
