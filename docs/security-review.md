# Security and privacy review · 2026-10-23

Scope: current tracked source, research reports, dependency lockfile, local web
API, upload handling, model-loading boundaries, and commit metadata. This is a
source and dependency review with regression tests, not an external penetration
test or a guarantee that all vulnerabilities or personal data have been found.

## Remediated

- Restricted accepted HTTP Host values to loopback hosts to reduce DNS-rebinding
  exposure. The launcher continues to bind only to 127.0.0.1.
- Enforced an actual streamed request-body ceiling before multipart parsing;
  the limit does not depend on Content-Length. Multipart uploads are closed
  after reading, including invalid file selections.
- Added no-store, anti-framing, content-type, referrer, and content-security
  policies. Inline scripts and external script/connect destinations are blocked;
  inline styles remain permitted for the existing interface.
- Replaced raw backend exception messages with generic API errors to avoid
  disclosing filesystem paths and local runtime details.
- Updated certifi to 2026.10.23 and Pillow to 13.0.0. Updated Streamlit to 1.66.1
  for compatibility. The pinned dependency audit changed from 36 known advisories
  in two packages to zero known advisories. Compatibility checking passed.
- Replaced personal absolute notice paths in two acquisition receipts with
  repository-relative paths. Required upstream model identifiers and license
  attribution are unchanged.
- Prevented future pilot acquisition receipts from storing absolute notice paths.

## Verification

378 Python tests and six frontend tests passed. Tests cover unknown Host
rejection, foreign-origin upload rejection, over-limit chunked uploads without
Content-Length, and response security headers. JavaScript syntax checks passed.
The lockfile audit reported no known vulnerabilities; it does not verify model
checkpoints, operating-system packages, or every separate research environment.

A pattern scan of 265 tracked and publication-candidate text files found no
personal workstation paths or supported credential patterns after cleanup.
The one email-shaped match is a deliberately invalid credential-in-URL test
fixture, not an actual contact address. Public repository account identifiers,
research source identifiers, and legally required attribution remain present.
The main-screen screenshot was visually reviewed during publication; this
review does not constitute exhaustive OCR of every older binary asset.

Bandit reported 47 findings: 8 high, 9 medium, and 30 low severity. The high
findings concern MD5/SHA-1 compatibility checks for publisher, Git, and FLAC
identifiers, rather than password storage. They were reviewed alongside stronger
SHA-256 checks and pinned-source handling; hashes are not a guarantee of a
publisher's identity. Medium findings include parameterized SQL with generated
placeholder lists, fixed-URL metadata acquisition, offline verified model loads,
and execution of hash-pinned research source. These remain trust boundaries,
particularly if acquisition code or pinned model artifacts are changed. The
scanner report has not been represented as a clean static-analysis result.

## Research receipt provenance after redaction

Original immutable receipts are retained locally under git-ignored
private/security-originals/. Published copies explicitly identify the original
receipt hash and their publication-only transformation. Dependent public receipt
references were rebound in dependency order so they refer to the sanitized
public versions. No model scores, audio identities, or historical audit outcomes
were changed. These transformations are not new detector or integrity-audit runs;
original code hashes and experimental results retain their historical meaning.
Existing local frozen acquisition specs refer to the original receipts and must
be migrated explicitly before any new acquisition run; fail-closed checks must
not be bypassed to resume an old specification against changed public receipts.

## Remaining limits

Published branch history was rewritten to remove personal workstation paths,
and commits use a generic maintainer identity. Git integrity checks passed after
the rewrite, and current file contents did not change. A rewrite does not erase
existing clones, backups, cached commit pages, or other retained copies.

The service has no authentication and is for a trusted local single operator.
Local software can still access it; loopback Host validation is not authentication.
Do not expose it to a network or deploy it as a shared service without access
controls, stronger resource/concurrency limits, and deployment-specific review.
Uploaded speech and analyst notes can themselves contain sensitive data. Local
processing and ignored storage reduce publication risk but do not anonymize them.
