# Expanded external audio set

The official ASVspoof5 evaluation archive now supplies 900 unscored recordings
from 600 declared speakers. This expands the 40-record pilot while preserving
every original source and its reserved partition.

| Reserved partition | Genuine recordings/groups | Synthetic recordings/groups | Total distinct speakers |
| --- | ---: | ---: | ---: |
| Calibration | 300 | 100 | 300 |
| Evaluation | 300 | 200 | 300 |

Some speakers supply both classes; they remain entirely in one partition.
There are zero declared-speaker overlaps between partitions. Speaker counts
are based on publisher metadata, not independent biometric verification.

The selection uses fixed SHA-ranked metadata before any detector scores. It
selects one recording per speaker/class and never replaces a selected file
based on its audio or a model result. The set covers all 16 attack labels A17
through A32 and all 11 codec labels plus uncompressed audio. Selected audio
totals 112,275,298 bytes; durations span 2.3360625–12.73425 seconds.

Acquisition extended each of ten authenticated 8 MiB pilot archive prefixes
to 96 MiB, a fixed additional transfer of 880 MiB. The 960 MiB total prefixes
contained 8,178 eligible complete members. Complete archive downloads were
avoided. Prefix sampling is not representative random sampling of the dataset.

The independent audit verified all ten prefix hashes, all 900 selected file
hashes against complete tar members, official protocol metadata, valid finite
mono 16 kHz decoding, decoded sample hashes, same-speaker codec parents,
reserved class minimums, and preservation of all 40 pilot reservations. It
found zero duplicate file or decoded PCM pairs and zero cross-partition
declared-speaker overlaps.

These are integrity and partition checks, not evidence of better detection.
No detector inference or threshold changes were performed. The reserved
evaluation set remains unscored. Training exposure, upstream model evaluation
exposure, cross-corpus perceptual/biometric overlap, and full archive publisher
checksums remain unverified. The WavLM publisher has already reported an
ASVspoof benchmark, so these sources cannot establish upstream-unseen testing.
Raw counts do not establish independent per-condition confirmation gates.

Publisher: [ASVspoof5 official record](https://zenodo.org/records/14498691).
The official README and license notices retained with the pilot also apply:
the database uses ODC-By and genuine recordings are described under CC BY 4.0;
individual recording rights require the publisher's complete notices. This
acquisition does not grant broader redistribution rights.

Receipts:

- [Frozen acquisition](asvspoof5-expanded-acquisition-v1.json)
- [Independent integrity audit](asvspoof5-expanded-qualification-v1.json)
- [Original pilot reservations](asvspoof5-pilot-reservation-v2.json)

Next: freeze the external stress-test protocol and candidate operating points
using calibration only, before scoring the reserved evaluation partition.
Keep unknown upstream exposure explicit in every resulting accuracy claim.
