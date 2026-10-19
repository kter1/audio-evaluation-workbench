"""Run real local-model diagnostics; never creates personal annotations."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.detectors import DETECTORS, load
from lab.forensics import analyze, save_report
from lab.learning import forensic_examples
from lab.store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detector', choices=list(DETECTORS), default='fusion')
    parser.add_argument('--db', default='data/study.sqlite')
    parser.add_argument('--output', default='data/analyses/reference_check.json')
    args = parser.parse_args()
    model = load(args.detector)
    rows = []
    with_db = Store(args.db)
    try:
        selected = forensic_examples(with_db)
        if len(selected) != 4:
            raise ValueError('Prepare four duration-qualified forensic examples first.')
        for item in selected:
            path = with_db.one('SELECT path FROM recordings WHERE id=?', (item['id'],))['path']
            report, x = analyze(Path(path).read_bytes(), model, robustness=True)
            save_report(report)
            rows.append({'id': item['id'], 'reference': item['label'],
                         'score': report['assessment']['synthetic_score'],
                         'assessment': report['assessment']['assessment'],
                         'seconds': report['runtime_seconds'],
                         'robustness_max_delta': max(abs(r['score_change']) for r in report['robustness'])})
            print(json.dumps(rows[-1]), flush=True)
            if len(rows) == 1:
                repeat, _ = analyze(Path(path).read_bytes(), model, robustness=False)
                assert np.allclose(report['assessment']['synthetic_score'], repeat['assessment']['synthetic_score'], rtol=1e-6, atol=1e-7)
        receipt = {'note': 'Four source-disjoint reference demonstrations; diagnostic only, training overlap unknown, not a generalization estimate.',
                   'model': model.manifest, 'results': rows}
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=2)+'\n')
        print('Receipt:', out)
    finally:
        with_db.close()


if __name__ == '__main__':
    main()
