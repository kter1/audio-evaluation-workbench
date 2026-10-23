"""Prepare source-verified, model-blind DEVELOPMENT speech review packets."""
from hashlib import sha256
import json
from pathlib import Path
import secrets
import shutil
import tempfile

import soundfile as sf

from lab.baseline_audit import checked_path


def build_packet(manifest_path, root, destination):
    root, destination = Path(root).resolve(), Path(destination).absolute()
    raw_manifest = Path(manifest_path).read_bytes()
    manifest = json.loads(raw_manifest)
    if manifest.get('schema_version') != 'method-benchmark-1.0' or manifest.get('partition') != 'development':
        raise ValueError('Review packets accept development data only')
    cases = manifest.get('cases', [])
    if not cases or any(c.get('partition', 'development') != 'development' for c in cases):
        raise ValueError('Nonempty development cases required')
    ids = [c['id'] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate case IDs')
    if destination.exists():
        raise ValueError('Destination must not exist')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.review-', dir=destination.parent))
    try:
        reviewer = stage/'reviewer'
        (reviewer/'audio').mkdir(parents=True)
        private = stage/'coordinator-only'
        private.mkdir()
        rows, mapping = [], []
        # Independent random order and opaque IDs avoid exposing source ordering.
        ordered = secrets.SystemRandom().sample(cases, len(cases))
        for index, case in enumerate(ordered, 1):
            source = checked_path(root, case['path'])
            raw = source.read_bytes()
            digest = sha256(raw).hexdigest()
            if digest != case['sha256']:
                raise ValueError('Source hash mismatch: '+case['id'])
            info = sf.info(source)
            if info.format != 'WAV':
                raise ValueError('Review packet currently requires WAV sources')
            opaque = f'clip-{index:04d}'
            target = reviewer/'audio'/f'{opaque}.wav'
            target.write_bytes(raw)
            rows.append({'id': opaque, 'path': f'audio/{opaque}.wav',
                         'sha256': digest, 'duration_seconds': info.duration,
                         'speech_intervals': None, 'status': 'pending', 'notes': ''})
            mapping.append({'review_id': opaque, 'source_id': case['id'],
                            'source_group': case['source_group'], 'sha256': digest})
        review = {'schema_version': 'speech-review-1.0', 'partition': 'development',
                  'reviewer': None, 'completed_at': None, 'cases': rows}
        (reviewer/'annotations.json').write_text(json.dumps(review, indent=2)+'\n')
        (reviewer/'README.md').write_text(
            '# Independent speech annotation\n\n'
            'Listen to every complete clip without consulting publisher labels or model outputs.\n'
            'Mark audible spoken language, including quiet speech; exclude music, breath-only\n'
            'sounds and nonverbal noise. Use seconds and nonoverlapping [start, end) intervals.\n'
            'Record ambiguous boundaries or vocalizations in notes. Do not guess: retain pending\n'
            'status where speech cannot be resolved. Null means unreviewed; [] means reviewed\n'
            'and no speech. Set status to reviewed only after listening to the entire clip.\n'
            'Enter your reviewer identifier and completion time. Preserve audio and hashes.\n'
            'Return annotations.json to the coordinator for comparison and adjudication.\n'
            'This packet is development-only and does not establish reviewer independence.\n')
        receipt = {'schema_version': 'speech-review-packet-1.0',
                   'manifest_sha256': sha256(raw_manifest).hexdigest(),
                   'mapping': mapping, 'case_count': len(rows),
                   'scope': 'All manifest cases; no selection by model error. Share reviewer/ only. '
                            'Neutral IDs do not remove spoken-content identity cues or establish independence. '
                            'Coordinator retains source licenses; redistribution requires separate clearance.'}
        (private/'mapping.json').write_text(json.dumps(receipt, indent=2)+'\n')
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return receipt
