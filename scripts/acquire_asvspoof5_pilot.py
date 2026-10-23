"""Acquire a score-blind bounded pilot from official ASVspoof5 archive prefixes."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import sys
import tarfile
from urllib.request import Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.research_downloads import allowed_url, download, open_approved

PREFIX_BYTES = 8*1024**2
SEED = 'asvspoof5-pilot-20261023:'


def prefix(url, archive_size, opener=open_approved):
    allowed_url(url)
    request = Request(url, headers={'Range': f'bytes=0-{PREFIX_BYTES-1}'})
    with opener(request, timeout=30) as response:
        allowed_url(response.geturl())
        if (response.status != 206 or response.headers.get('Content-Range') !=
                f'bytes 0-{PREFIX_BYTES-1}/{archive_size}'):
            raise ValueError('Publisher must honor the exact bounded archive range')
        raw = response.read(PREFIX_BYTES+1)
    if len(raw) != PREFIX_BYTES:
        raise ValueError('Archive range differs from byte budget')
    return raw


def members(raw):
    """Read only complete safe regular members; never extract archive paths."""
    offset = 0
    while offset+512 <= len(raw):
        header = raw[offset:offset+512]
        if header == bytes(512):
            break
        item = tarfile.TarInfo.frombuf(header, 'utf-8', 'strict')
        start = offset+512
        end = start+item.size
        if end > len(raw):
            break  # Expected incomplete final member in a byte-range prefix.
        if item.isreg() and re.fullmatch(r'flac_E_eval/E_\d{10}\.flac', item.name):
            yield item.name, raw[start:end], start
        offset = start+((item.size+511)//512)*512


def choose(entries):
    """One record per declared speaker, equal classes, no audio/score inspection."""
    selected, used = [], set()
    ordered = sorted(entries, key=lambda r: sha256((SEED+r['id']).encode()).hexdigest())
    for label in ('human', 'synthetic'):
        part = []
        for row in ordered:
            if row['label'] == label and row['speaker'] not in used:
                part.append(row)
                used.add(row['speaker'])
                if len(part) == 20:
                    break
        if len(part) != 20:
            raise ValueError('Pilot prefixes do not support 20 distinct speakers per class')
        selected.extend(part)
    groups = sorted(used, key=lambda g: sha256((SEED+'partition:'+g).encode()).hexdigest())
    calibration = set(groups[:20])
    return [{**r, 'partition': 'calibration_reserved' if r['speaker'] in calibration else 'evaluation_reserved'}
            for r in selected]


def main():
    directory = ROOT/'data/research-corpora/asvspoof5-pilot'
    directory.mkdir(parents=True, exist_ok=True)
    output = ROOT/'docs/asvspoof5-pilot-acquisition-v1.json'
    if output.exists():
        raise ValueError('Pilot receipt already exists; do not overwrite it')
    with open_approved('https://zenodo.org/api/records/14498691', 30) as response:
        metadata_raw = response.read(1024**2+1)
    if len(metadata_raw) > 1024**2:
        raise ValueError('Publisher metadata exceeds budget')
    metadata = json.loads(metadata_raw)
    archives = sorted([f for f in metadata['files'] if re.fullmatch('flac_E_a[a-j].tar', f['key'])], key=lambda f:f['key'])
    if len(archives) != 10:
        raise ValueError('Require ten declared official evaluation archives')
    spec = {'publisher_record': 'https://zenodo.org/records/14498691',
        'metadata_sha256': sha256(metadata_raw).hexdigest(), 'prefix_bytes_per_archive': PREFIX_BYTES,
        'maximum_audio_transfer_bytes': 10*PREFIX_BYTES, 'target': {'human':20,'synthetic':20},
        'selection': 'First 20 per class by SHA256(seed+ID), one declared speaker per record, human class selected first; metadata only',
        'seed': SEED, 'archives': [{'name':f['key'],'size':f['size'],'publisher_checksum':f['checksum'],
            'url':f"https://zenodo.org/records/14498691/files/{f['key']}?download=1&pilot_prefix={PREFIX_BYTES}"} for f in archives],
        'partition_policy': 'SHA256(seed+partition:+speaker) first20 calibration_reserved, rest evaluation_reserved; no inference or threshold selection',
        'scope': 'New-to-project acquisition pilot only. Training exposure and cross-corpus speaker identities unresolved; not qualified independent locked validation. Full-archive publisher MD5 cannot be verified from prefixes.'}
    spec_path = directory/'selection-spec.json'
    text = json.dumps(spec, indent=2)+'\n'
    if spec_path.exists() and spec_path.read_text() != text:
        raise ValueError('Frozen selection specification differs')
    if not spec_path.exists():
        spec_path.write_text(text)
    protocol_archive = ROOT/'data/research-corpora/asvspoof5/ASVspoof5_protocols.tar.gz'
    if sha256(protocol_archive.read_bytes()).hexdigest() != 'ab4505dcce14306d99e60f4255dfdd0ee9988d225e7194698da219dda690ec81':
        raise ValueError('Official protocol archive hash mismatch')
    with tarfile.open(protocol_archive) as archive:
        protocol = archive.extractfile('ASVspoof5.eval.track_1.tsv').read()
    indexed = {}
    for line in protocol.decode().splitlines():
        values = line.split()
        indexed[values[1]] = values
    def acquire(entry):
        path = directory/(entry['name']+'.prefix')
        if path.exists():
            raise ValueError('Unexpected existing range file; use a fresh acquisition directory')
        raw = prefix(entry['url'], entry['size'])
        with path.open('xb') as stream:
            stream.write(raw)
        return entry, raw
    entries, payloads, receipts = [], {}, []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for archive, raw in pool.map(acquire, spec['archives']):
            receipts.append({'archive':archive['name'], 'bytes':len(raw),'sha256':sha256(raw).hexdigest()})
            for name, audio, offset in members(raw):
                record = Path(name).stem
                if record not in indexed:
                    continue
                v = indexed[record]
                if v[8] not in ('bonafide','spoof'):
                    raise ValueError('Unknown publisher authenticity label')
                row = {'id':record,'speaker':v[0],'group':'asvspoof5:'+v[0],
                    'label':'human' if v[8]=='bonafide' else 'synthetic',
                    'codec':v[3], 'codec_seed':v[5], 'attack':v[7],
                    'archive':archive['name'],'member':name,'member_offset':offset,
                    'bytes':len(audio),'sha256':sha256(audio).hexdigest()}
                entries.append(row)
                payloads[record] = audio
            print('Indexed bounded prefix:', archive['name'], flush=True)
    selected = choose(entries)
    import soundfile as sf
    for row in selected:
        audio = payloads[row['id']]
        info = sf.info(io.BytesIO(audio))
        if info.samplerate != 16000 or info.channels != 1 or info.frames <= 0:
            raise ValueError('Selected publisher audio has unexpected format')
        path = directory/'audio'/(row['id']+'.flac')
        path.parent.mkdir(exist_ok=True)
        with path.open('xb') as stream:
            stream.write(audio)
        row.update({'path':str(path.relative_to(ROOT)), 'duration_seconds':info.duration})
    notices = []
    for name, md5 in [('README.txt','784002b8449ba1eeecc042bfad285de7'),('LICENSE.txt','9cbefc63251e8cbc8eb22e62cd6fe584')]:
        notices.append(download(f'https://zenodo.org/records/14498691/files/{name}?download=1',directory/name,
                                maximum=32768,publisher_md5=md5))
        notices[-1]['path'] = str((directory/name).relative_to(ROOT))
    result = {'spec':spec,'archive_prefixes':receipts,'eligible_complete_members':len(entries),
        'files':selected,'declared_speaker_groups':len({r['speaker'] for r in selected}),
        'protocol_sha256':sha256(protocol).hexdigest(),'notices':notices,
        'license': 'ODC-By database; CC-BY-4.0 bona fide contents, per official README; retain original notices',
        'detector_inference_run':False,'qualified_for_confirmatory_evaluation':False,
        'integrity':'HTTPS official immutable record, exact Content-Range, tar header checksums, complete-member SHA-256, soundfile headers. Full archive MD5 not verified.',
        'limitations':['Only bounded archive prefixes; not a representative full benchmark.',
                        'Declared speaker groups are not cross-corpus biometric independence.',
                        'Codec ancestors and model training exposure require qualification before calibration or final testing.']}
    with output.open('x') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Acquired40 unscored recordings /40 declared speakers. No model or app changes.')


if __name__ == '__main__':
    main()
