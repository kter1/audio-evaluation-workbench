"""Optional offline transcription of the fixed demonstration; requires the pinned cached checkpoint."""
from pathlib import Path
from hashlib import sha256
import json
import torch
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
root=Path(__file__).resolve().parents[1]
revision='911407f4214e0e1d82085af863093ec0b66f9cd6'
directory=root/'data/hf-cache/hub/models--openai--whisper-base.en/snapshots'/revision
CONFIG_HASHES = {'added_tokens.json': '560be47bea388757f8d4cc185c5d82067426cbb6361e38016dd90ddc01ab203a', 'tokenizer_config.json': '14f84bdf4b9ecdbd4738ddc81c17a1baedfc02bb93c6e049c951e15a1b40b70d', 'special_tokens_map.json': '014f8f802366ed818919550be0ad9e35907327cb9e142e8aaa102420f460bda8', 'config.json': '160c1df40a60d4ef5a6014d536fd845c33aabfd1f318458ac7f87cfe91ebbe76', 'tokenizer.json': '5eb60cec1e77aeeb6869a2bb5a8e01a84c3fe5d072d75369343021fe6f5310d0', 'generation_config.json': 'c5750f05d94777579e00ce26ef65e5d87c108439f90e3ac519df2587b9d5d41f', 'normalizer.json': 'bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd', 'merges.txt': '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5', 'vocab.json': '3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7', 'preprocessor_config.json': '9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666'}
for name, expected in CONFIG_HASHES.items():
    if sha256((directory/name).read_bytes()).hexdigest() != expected:
        raise ValueError('Cached transcription configuration differs from the pinned files')
raw=(directory/'model.safetensors').read_bytes()
if sha256(raw).hexdigest() != 'd4dd5542fd6a1d35639e21384238f3bfe6c557c849d392b5905d33ee29e71db5':
    raise ValueError('Transcription checkpoint identity mismatch')
torch.set_num_threads(4)
processor=WhisperProcessor.from_pretrained(str(directory),local_files_only=True,trust_remote_code=False)
model=WhisperForConditionalGeneration.from_pretrained(str(directory),local_files_only=True,trust_remote_code=False,use_safetensors=True,attn_implementation='eager')
asr=pipeline('automatic-speech-recognition',model=model,tokenizer=processor.tokenizer,feature_extractor=processor.feature_extractor,device='cpu')
sample=root/'studio/dist/samples/annotation-demo.wav'
audio,rate=sf.read(sample,dtype='float32')
output=asr({'raw':audio,'sampling_rate':rate},return_timestamps='word',generate_kwargs={'do_sample':False,'num_beams':1})
result={'model':'openai/whisper-base.en','revision':revision,'weights_sha256':sha256(raw).hexdigest(),'sample_sha256':sha256(sample.read_bytes()).hexdigest(),'configuration_sha256':CONFIG_HASHES,'method':'Local deterministic CPU transcription with model-estimated word timestamps; no human verification or authenticity inference.','text':output['text'],'words':[{'text':c['text'].strip(),'start':c['timestamp'][0],'end':c['timestamp'][1]} for c in output['chunks']]}
(root/'studio/dist/samples/annotation-transcript.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(result['text']);print('Words:',len(result['words']))
