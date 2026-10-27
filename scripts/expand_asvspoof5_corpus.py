"""Expand official, unscored ASVspoof5 sources under a fixed range budget.

Preserve the pilot reservations; select by metadata only, never model scores.
"""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from urllib.request import Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.baseline_audit import checked_path
from lab.model_acquisition import file_sha256
from lab.research_downloads import allowed_url, open_approved
from scripts.acquire_asvspoof5_pilot import members

END = 96*1024**2
SEED = 'asvspoof5-expansion-20261023:'
OUTPUT = ROOT/'docs/asvspoof5-expanded-acquisition-v1.json'
DIRECTORY = ROOT/'data/research-corpora/asvspoof5-expanded'


def rank(value):
    return sha256((SEED+value).encode()).hexdigest()


def reserve(entries, pilot):
    """Preserve pilot groups and satisfy fixed class/group minimums without scores."""
    if len({r['id'] for r in entries}) != len(entries):
        raise ValueError('Duplicate source identity')
    available = {label:{r['speaker'] for r in entries if r['label']==label}
                 for label in ('human','synthetic')}
    h, s = available['human'], available['synthetic']
    roles = {r['speaker']:r['partition'] for r in pilot}
    calibration = {g for g,p in roles.items() if p=='calibration_reserved'}
    evaluation = {g for g,p in roles.items() if p=='evaluation_reserved'}
    if calibration & evaluation:
        raise ValueError('Pilot group crosses reserved partitions')
    def fill(target, excluded, candidates, relevant, minimum):
        for group in sorted(candidates-excluded-target, key=rank):
            if len(target & relevant) >= minimum:
                break
            target.add(group)
        if len(target & relevant) < minimum:
            raise ValueError('Fixed source-group minimum unavailable within acquisition budget')
    fill(evaluation, calibration, h&s, s, 200)
    fill(evaluation, calibration, h-s, h, 300)
    fill(calibration, evaluation, h&s, s, 100)
    fill(calibration, evaluation, h-s, h, 300)
    existing = {(r['speaker'],r['label']):r['id'] for r in pilot}
    by_id = {r['id']:r for r in entries}
    chosen = []
    for partition, groups in (('calibration_reserved',calibration),('evaluation_reserved',evaluation)):
        for speaker in sorted(groups):
            for label in ('human','synthetic'):
                possible = [r for r in entries if r['speaker']==speaker and r['label']==label]
                if not possible:
                    continue
                old = existing.get((speaker,label))
                if old is not None:
                    row = by_id.get(old)
                    if row is None or row not in possible:
                        raise ValueError('Original pilot source is missing')
                else:
                    row = min(possible,key=lambda r:rank('record:'+r['id']))
                chosen.append({**row,'partition':partition})
    if not {r['id'] for r in pilot} <= {r['id'] for r in chosen}:
        raise ValueError('Pilot reservations must be preserved')
    return chosen


def extend(entry, cached):
    """Fetch exactly bytes after the authenticated pilot prefix; retain partials nowhere."""
    pilot_dir = ROOT/'data/research-corpora/asvspoof5-pilot'
    base = pilot_dir/(entry['name']+'.prefix')
    if base.stat().st_size != cached['bytes'] or file_sha256(base) != cached['sha256']:
        raise ValueError('Pilot prefix identity mismatch')
    start = cached['bytes']
    target = DIRECTORY/(entry['name']+'.prefix96')
    receipt_path = DIRECTORY/(entry['name']+'.range.json')
    if target.exists() or receipt_path.exists():
        if not (target.exists() and receipt_path.exists()):
            raise ValueError('Incomplete cached acquisition; do not trust it')
        receipt = json.loads(receipt_path.read_text())
        if (receipt['bytes'] != END or target.stat().st_size != END
                or receipt['sha256'] != file_sha256(target)
                or receipt['pilot_sha256'] != cached['sha256']):
            raise ValueError('Expanded cache identity mismatch')
        return receipt
    url = entry['url'].split('?')[0]+f'?download=1&expansion_end={END}'
    allowed_url(url)
    request = Request(url,headers={'Range':f'bytes={start}-{END-1}'})
    temporary = None
    try:
        with open_approved(request,30) as response:
            allowed_url(response.geturl())
            if (response.status != 206 or response.headers.get('Content-Range') !=
                    f'bytes {start}-{END-1}/{entry["size"]}'):
                raise ValueError('Exact bounded range was not honored')
            with tempfile.NamedTemporaryFile(dir=DIRECTORY,prefix='.range-',delete=False) as stream:
                temporary = Path(stream.name)
                digest = sha256()
                with base.open('rb') as original:
                    for chunk in iter(lambda:original.read(1024**2),b''):
                        stream.write(chunk);digest.update(chunk)
                received = 0
                while True:
                    chunk = response.read(min(1024**2,END-start-received+1))
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > END-start:
                        raise ValueError('Expanded range exceeds fixed byte budget')
                    stream.write(chunk);digest.update(chunk)
            if received != END-start:
                raise ValueError('Expanded range is incomplete')
        receipt = {'archive':entry['name'],'url':url,'bytes':END,'additional_bytes':received,
                   'range_start':start,'range_end_inclusive':END-1,
                   'sha256':digest.hexdigest(),'pilot_sha256':cached['sha256']}
        os.link(temporary,target)
        with receipt_path.open('x') as stream:
            stream.write(json.dumps(receipt,indent=2)+'\n')
        return receipt
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    if OUTPUT.exists():
        raise ValueError('Existing expanded receipt is immutable; run the audit instead')
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    source_path = ROOT/'docs/asvspoof5-pilot-acquisition-v1.json'
    reservation_path = ROOT/'docs/asvspoof5-pilot-reservation-v2.json'
    source_raw, reservation_raw = source_path.read_bytes(),reservation_path.read_bytes()
    source, reservation = json.loads(source_raw),json.loads(reservation_raw)
    if reservation['source_acquisition_sha256'] != sha256(source_raw).hexdigest():
        raise ValueError('Pilot reservation does not match acquisition')
    archives = source['spec']['archives']
    if len(archives) != 10 or any(not 0 < f['size'] > END for f in archives):
        raise ValueError('Require ten official larger archives')
    spec = {'version':'1.0','pilot_acquisition_sha256':sha256(source_raw).hexdigest(),
        'pilot_reservation_sha256':sha256(reservation_raw).hexdigest(),'seed':SEED,
        'prefix_bytes_per_archive':END,'additional_transfer_max_bytes':10*(END-8*1024**2),
        'minimum_groups_by_class':{'calibration_reserved':{'human':300,'synthetic':100},
                                   'evaluation_reserved':{'human':300,'synthetic':200}},
        'selection':'Preserve pilot records/roles; SHA-ranked matched speakers fill evaluation synthetic first, then human-only groups; calibration analogous. One record per speaker/class, old pilot record preferred, otherwise SHA(seed+record:+ID). No fallback after scoring.',
        'detector_inference_permitted':False,'scope':'Unscored acquisition expansion, not confirmation. Unknown model-training and cross-corpus identity overlap; prefix sampling is not representative.'}
    spec_path = DIRECTORY/'selection-spec.json'
    text = json.dumps(spec,indent=2)+'\n'
    if spec_path.exists() and spec_path.read_text() != text:
        raise ValueError('Frozen acquisition specification differs')
    if not spec_path.exists():
        spec_path.write_text(text)
    cached = {r['archive']:r for r in source['archive_prefixes']}
    receipts = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for receipt in pool.map(lambda a:extend(a,cached[a['name']]),archives):
            receipts.append(receipt)
            print('Verified expanded archive prefix:',receipt['archive'],flush=True)
    protocol_path = ROOT/'data/research-corpora/asvspoof5/ASVspoof5_protocols.tar.gz'
    if file_sha256(protocol_path) != 'ab4505dcce14306d99e60f4255dfdd0ee9988d225e7194698da219dda690ec81':
        raise ValueError('Official protocol archive differs')
    with tarfile.open(protocol_path) as archive:
        protocol = archive.extractfile('ASVspoof5.eval.track_1.tsv').read()
    if sha256(protocol).hexdigest() != source['protocol_sha256']:
        raise ValueError('Official evaluation protocol differs')
    metadata = {v[1]:v for v in (line.split() for line in protocol.decode().splitlines())}
    entries, seen = [], set()
    for receipt in receipts:
        for name, audio, offset in members((DIRECTORY/(receipt['archive']+'.prefix96')).read_bytes()):
            record = Path(name).stem
            if record not in metadata:
                continue
            if record in seen:
                raise ValueError('Duplicate publisher source identity')
            seen.add(record)
            v = metadata[record]
            if v[8] not in ('bonafide','spoof'):
                raise ValueError('Invalid publisher label')
            if v[5] != '-' and (v[5] not in metadata or metadata[v[5]][0] != v[0]):
                raise ValueError('Unresolved/cross-speaker codec relationship')
            entries.append({'id':record,'speaker':v[0],'group':'asvspoof5:'+v[0],
                'label':'human' if v[8]=='bonafide' else 'synthetic','codec':v[3],
                'codec_seed':v[5],'attack':v[7],'archive':receipt['archive'],
                'member':name,'member_offset':offset,'bytes':len(audio),'sha256':sha256(audio).hexdigest()})
    selected = reserve(entries,reservation['records'])
    import numpy as np
    import soundfile as sf
    import io
    hashes, pcm_hashes = set(),set()
    for row in selected:
        with (DIRECTORY/(row['archive']+'.prefix96')).open('rb') as stream:
            stream.seek(row['member_offset']);audio=stream.read(row['bytes'])
        if sha256(audio).hexdigest() != row['sha256']:
            raise ValueError('Selected member changed after selection')
        samples,rate=sf.read(io.BytesIO(audio),dtype='float32')
        if rate != 16000 or samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
            raise ValueError('Selected source cannot be decoded; no replacement allowed')
        pcm=sha256(samples.astype('<f4').tobytes()).hexdigest()
        if row['sha256'] in hashes or pcm in pcm_hashes:
            raise ValueError('Duplicate bytes/decoded samples require grouping review')
        hashes.add(row['sha256']);pcm_hashes.add(pcm)
        target=DIRECTORY/'audio'/(row['id']+'.flac');target.parent.mkdir(exist_ok=True)
        if target.exists():
            if file_sha256(target) != row['sha256']:
                raise ValueError('Existing selected audio differs')
        else:
            with target.open('xb') as stream:stream.write(audio)
        row.update({'path':str(target.relative_to(ROOT)),'decoded_pcm_sha256':pcm,
                    'duration_seconds':len(samples)/rate,'samples':len(samples)})
    summaries={p:{label:sum(r['partition']==p and r['label']==label for r in selected)
        for label in ('human','synthetic')} for p in ('calibration_reserved','evaluation_reserved')}
    result={'spec':spec,'prefixes':receipts,'eligible_complete_records':len(entries),
        'files':selected,'partitions':summaries,'declared_speaker_groups':len({r['speaker'] for r in selected}),
        'protocol_sha256':sha256(protocol).hexdigest(),'detector_inference_run':False,
        'qualified_confirmatory':False,'code_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'notices':source['notices'],'license':source['license'],
        'limitations':['Full archive publisher MD5 unverified; exact ranges, tar headers and observed SHA-256 only.',
                      'Same-speaker codec parents grouped; cloning/source identity beyond publisher metadata unresolved.',
                      'No claim of absence from pretraining, training or upstream model evaluation.',
                      'Counts do not establish per-condition independent-group gates; no detector or threshold changes.']}
    with OUTPUT.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Unscored expanded sources:',len(selected),'groups:',result['declared_speaker_groups'],summaries)


if __name__ == '__main__':
    main()
