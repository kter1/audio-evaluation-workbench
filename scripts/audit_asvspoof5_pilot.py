"""Verify the unscored pilot and disclose unresolved independence constraints."""
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
from lab.corpus_inventory import inventory
from scripts.acquire_asvspoof5_pilot import members


def main():
    source = ROOT/'docs/asvspoof5-pilot-acquisition-v1.json'
    raw = source.read_bytes()
    acquisition = json.loads(raw)
    files = acquisition['files']
    if len(files) != 40 or len({r['id'] for r in files}) != 40 or len({r['speaker'] for r in files}) != 40:
        raise ValueError('Require forty distinct selected records and declared speakers')
    directory = ROOT/'data/research-corpora/asvspoof5-pilot'
    archived = {}
    for entry in acquisition['archive_prefixes']:
        data = (directory/(entry['archive']+'.prefix')).read_bytes()
        if len(data) != entry['bytes'] or sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Frozen prefix hash mismatch')
        for name, audio, offset in members(data):
            key = (entry['archive'], name)
            if key in archived:
                raise ValueError('Duplicate archive member identity')
            archived[key] = (sha256(audio).hexdigest(), offset)
    with tarfile.open(ROOT/'data/research-corpora/asvspoof5/ASVspoof5_protocols.tar.gz') as archive:
        protocol = archive.extractfile('ASVspoof5.eval.track_1.tsv').read()
    if sha256(protocol).hexdigest() != acquisition['protocol_sha256']:
        raise ValueError('Protocol identity mismatch')
    metadata = {v[1]:v for v in (line.split() for line in protocol.decode().splitlines())}
    previous = inventory(ROOT, include_records=True)
    prior_tokens = {t for r in previous['identity_records'] for t in r['identity_tokens']}
    rows, hashes, pcm_hashes = [], set(), set()
    for row in files:
        path = checked_path(ROOT, row['path'])
        blob = path.read_bytes()
        h = sha256(blob).hexdigest()
        if (h != row['sha256'] or len(blob) != row['bytes']
                or archived[(row['archive'], row['member'])] != (h, row['member_offset'])):
            raise ValueError('Selected audio does not match complete frozen member')
        v = metadata[row['id']]
        if (v[0] != row['speaker'] or v[3] != row['codec'] or v[5] != row['codec_seed']
                or v[7] != row['attack'] or ('human' if v[8]=='bonafide' else 'synthetic') != row['label']):
            raise ValueError('Selected metadata differs from official protocol')
        audio, rate = sf.read(path, dtype='float32')
        if rate != 16000 or audio.ndim != 1 or not np.isfinite(audio).all() or not len(audio):
            raise ValueError('Invalid decoded pilot audio')
        pcm = sha256(audio.astype('<f4').tobytes()).hexdigest()
        if h in hashes or pcm in pcm_hashes:
            raise ValueError('Duplicate source bytes or decoded audio in pilot')
        hashes.add(h); pcm_hashes.add(pcm)
        seed = row['codec_seed']
        parent = metadata.get(seed) if seed != '-' else None
        if seed != '-' and (parent is None or parent[0] != row['speaker']):
            raise ValueError('Unresolved or cross-speaker codec parent prevents reserved split')
        rows.append({'id':row['id'],'partition':row['partition'],'speaker':row['speaker'],
            'source_sha256':h,'decoded_pcm_sha256':pcm,'samples':len(audio),
            'codec_parent':seed,'parent_speaker':parent[0] if parent else None,
            'prior_registered_byte_overlap': 'audio:'+h in prior_tokens})
    if any(r['prior_registered_byte_overlap'] for r in rows):
        raise ValueError('Pilot overlaps registered prior source bytes')
    calibration = {r['speaker'] for r in rows if r['partition']=='calibration_reserved'}
    evaluation = {r['speaker'] for r in rows if r['partition']=='evaluation_reserved'}
    if len(calibration) != 20 or len(evaluation) != 20 or calibration & evaluation:
        raise ValueError('Reserved partitions are not declared-speaker disjoint')
    result = {'acquisition_sha256':sha256(raw).hexdigest(),'audio_files_verified':40,
        'archive_prefixes_verified':10,'calibration_declared_speakers':20,'evaluation_declared_speakers':20,
        'registered_prior_byte_overlaps':0,'detector_inference_run':False,'qualified_confirmatory':False,
        'records':rows,'code_sha256':{p:sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ('scripts/acquire_asvspoof5_pilot.py','scripts/audit_asvspoof5_pilot.py','lab/corpus_inventory.py')},
        'scope':'Integrity and declared-identity qualification only. No scores computed. '
                'Full archive MD5, perceptual cross-corpus duplicates, biometric speaker overlap and model-training exposure remain unverified. '
                'Sample counts fall below confirmatory gates. Reserved evaluation remains unscored; it is not a qualified independent locked set.'}
    with (ROOT/'docs/asvspoof5-pilot-qualification-v1.json').open('x') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('40 decoded files /10 prefix hashes /20+20 reserved speakers verified; zero registered byte overlap; no detector run.')


if __name__ == '__main__':
    main()
