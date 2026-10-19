"""Count-based detector evaluation with denominators and a Wilson interval."""
from math import sqrt
import re


def wilson(success, n, z=1.959963984540054):
    if n <= 0: return None
    p=success/n; den=1+z*z/n
    centre=(p+z*z/(2*n))/den
    half=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,centre-half),min(1.,centre+half)]


def summarize(rows, threshold=.5):
    if not rows: raise ValueError('Evaluation needs scored cases.')
    if any(r['label'] not in ('human','synthetic') or not 0<=r['score']<=1 for r in rows):
        raise ValueError('Invalid label or score.')
    human=[r for r in rows if r['label']=='human']; fake=[r for r in rows if r['label']=='synthetic']
    fp=sum(r['score']>=threshold for r in human); fn=sum(r['score']<threshold for r in fake)
    correct=len(rows)-fp-fn
    decided=[r for r in rows if r['assessment']!='Inconclusive']
    dc=sum((r['score']>=threshold)==(r['label']=='synthetic') for r in decided)
    auc=sum((s['score']>h['score'])+.5*(s['score']==h['score']) for s in fake for h in human)/(len(fake)*len(human)) if human and fake else None
    return {'n':len(rows),'human':len(human),'synthetic':len(fake),'correct':correct,
            'false_positive':fp,'false_negative':fn,'accuracy':correct/len(rows),
            'accuracy_wilson95':wilson(correct,len(rows)),
            'balanced_accuracy':1-(fp/len(human)+fn/len(fake))/2 if human and fake else None,
            'roc_auc':auc,'abstained':len(rows)-len(decided),'decided':len(decided),
            'correct_among_decided':dc,'accuracy_among_decided':dc/len(decided) if decided else None}


def detection_rate(rows, threshold=.5):
    """Single-class cells: share flagged synthetic (fakes caught, or real false alarms)."""
    flagged=sum(r['score']>=threshold for r in rows)
    return {'n':len(rows),'flagged_synthetic':flagged,'rate':flagged/len(rows) if rows else None,
            'wilson95':wilson(flagged,len(rows))}


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def wer(reference, hypothesis):
    """Word error rate: word-level Levenshtein distance over reference length."""
    ref, hyp = words(reference), words(hypothesis)
    if not ref: raise ValueError('Reference transcript is empty.')
    row=list(range(len(hyp)+1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j]+1, row[j-1]+1, prev+(r!=h))
    return row[-1]/len(ref)
