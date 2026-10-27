"""Independently verify expanded source integrity and reserved speaker separation."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.baseline_audit import checked_path
from lab.model_acquisition import file_sha256
from scripts.acquire_asvspoof5_pilot import members


def check_reservations(rows, pilot):
    roles, ids, counts = {}, {}, Counter()
    for row in rows:
        speaker, partition, label = row['speaker'], row['partition'], row['label']
        if partition not in ('calibration_reserved', 'evaluation_reserved') or label not in ('human', 'synthetic'):
            raise ValueError('Unknown reserved role or class')
        if row['id'] in ids or (speaker, label) in counts:
            raise ValueError('Duplicate source or speaker/class')
        if speaker in roles and roles[speaker] != partition:
            raise ValueError('Speaker crosses reserved partitions')
        roles[speaker] = partition
        ids[row['id']] = row
        counts[speaker, label] += 1
    for old in pilot:
        new = ids.get(old['id'])
        if new is None or any(new[k] != old[k] for k in ('speaker', 'label', 'partition')):
            raise ValueError('Pilot reservation changed')
        if new['sha256'] != old['source_sha256']:
            raise ValueError('Pilot source bytes changed')
    totals = {p: {label: sum(r['partition'] == p and r['label'] == label for r in rows)
                  for label in ('human', 'synthetic')}
              for p in ('calibration_reserved', 'evaluation_reserved')}
    for p, synthetic in (('calibration_reserved', 100), ('evaluation_reserved', 200)):
        if totals[p]['human'] < 300 or totals[p]['synthetic'] < synthetic:
            raise ValueError('Reserved class/group minimum unmet')
    return totals


def main():
    output = ROOT/'docs/asvspoof5-expanded-qualification-v1.json'
    if output.exists():
        raise ValueError('Qualification receipt already exists')
    path = ROOT/'docs/asvspoof5-expanded-acquisition-v1.json'
    raw = path.read_bytes()
    source = json.loads(raw)
    reservation_raw = (ROOT/'docs/asvspoof5-pilot-reservation-v2.json').read_bytes()
    if sha256(reservation_raw).hexdigest() != source['spec']['pilot_reservation_sha256']:
        raise ValueError('Reservation receipt identity mismatch')
    totals = check_reservations(source['files'], json.loads(reservation_raw)['records'])
    directory = ROOT/'data/research-corpora/asvspoof5-expanded'
    archived = {}
    if len(source['prefixes']) != 10 or len({p['archive'] for p in source['prefixes']}) != 10:
        raise ValueError('Require ten distinct archive prefixes')
    for entry in source['prefixes']:
        prefix = checked_path(directory, entry['archive']+'.prefix96')
        if prefix.stat().st_size != entry['bytes'] or file_sha256(prefix) != entry['sha256']:
            raise ValueError('Archive prefix identity mismatch')
        for name, audio, offset in members(prefix.read_bytes()):
            key = (entry['archive'], name)
            if key in archived:
                raise ValueError('Duplicate archive member')
            archived[key] = (sha256(audio).hexdigest(), offset, len(audio))
    with tarfile.open(ROOT/'data/research-corpora/asvspoof5/ASVspoof5_protocols.tar.gz') as archive:
        protocol = archive.extractfile('ASVspoof5.eval.track_1.tsv').read()
    if sha256(protocol).hexdigest() != source['protocol_sha256']:
        raise ValueError('Protocol identity mismatch')
    metadata = {v[1]: v for v in (line.split() for line in protocol.decode().splitlines())}
    hashes, pcm_hashes, durations = set(), set(), []
    for row in source['files']:
        audio_path = checked_path(ROOT, row['path'])
        digest = file_sha256(audio_path)
        if (digest, row['member_offset'], audio_path.stat().st_size) != archived[row['archive'], row['member']]:
            raise ValueError('Audio differs from frozen archive member')
        if digest != row['sha256'] or audio_path.stat().st_size != row['bytes']:
            raise ValueError('Audio differs from selected source')
        v = metadata[row['id']]
        expected = (v[0], v[3], v[5], v[7], 'human' if v[8] == 'bonafide' else 'synthetic')
        if expected != tuple(row[k] for k in ('speaker', 'codec', 'codec_seed', 'attack', 'label')):
            raise ValueError('Source metadata differs from official protocol')
        if v[5] != '-' and (v[5] not in metadata or metadata[v[5]][0] != v[0]):
            raise ValueError('Unresolved codec parent identity')
        audio, rate = sf.read(audio_path, dtype='float32')
        if rate != 16000 or audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError('Invalid decoded source')
        pcm = sha256(audio.astype('<f4').tobytes()).hexdigest()
        if pcm != row['decoded_pcm_sha256'] or digest in hashes or pcm in pcm_hashes:
            raise ValueError('Changed or duplicate decoded source')
        if row['samples'] != len(audio) or row['duration_seconds'] != len(audio)/rate:
            raise ValueError('Decoded duration differs')
        hashes.add(digest)
        pcm_hashes.add(pcm)
        durations.append(len(audio)/rate)
    result = {'acquisition_sha256': sha256(raw).hexdigest(), 'audio_files_verified': len(hashes),
              'archive_prefixes_verified': 10, 'partitions': totals,
              'declared_speaker_groups': len({r['speaker'] for r in source['files']}),
              'pilot_sources_preserved': 40, 'cross_partition_speaker_overlap': 0,
              'duplicate_source_or_pcm_pairs': 0, 'duration_seconds_range': [min(durations), max(durations)],
              'detector_inference_run': False, 'qualified_confirmatory': False,
              'code_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Integrity and declared speaker separation only. Training exposure, biometric identity, '
                       'cross-corpus perceptual overlap, representative sampling and full archive publisher '
                       'checksums remain unverified. Counts alone do not qualify an independent detector benchmark.'}
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('scope', 'code_sha256')}, indent=2))


if __name__ == '__main__':
    main()
