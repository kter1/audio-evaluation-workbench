"""Descriptive local timing summaries; no hardware-independent performance claim."""
import math

import numpy as np


def peak_rss_gib(value, system):
    if system not in ('Darwin', 'Linux'):
        raise ValueError('Peak RSS units supported only for macOS and Linux')
    return value*(1 if system == 'Darwin' else 1024)/1024**3


def timing_summary(rows):
    if not rows or any(not math.isfinite(r[k]) or r[k] <= 0 for r in rows
                       for k in ('audio_seconds', 'wall_seconds')):
        raise ValueError('Positive finite audio durations and timings required')
    total_audio = sum(r['audio_seconds'] for r in rows)
    total_wall = sum(r['wall_seconds'] for r in rows)
    groups = {}
    for row in rows:
        duration = row['audio_seconds']
        name = '<5s' if duration < 5 else '5-15s' if duration < 15 else '>=15s'
        groups.setdefault(name, []).append(row['wall_seconds'])
    return {'timed_calls': len(rows), 'audio_seconds_including_repeats': total_audio,
            'wall_seconds': total_wall, 'realtime_factor': total_wall/total_audio,
            'latency_p50_seconds': float(np.quantile([r['wall_seconds'] for r in rows], .5)),
            'latency_p95_seconds': float(np.quantile([r['wall_seconds'] for r in rows], .95)),
            'duration_bins': {name: {'calls': len(times), 'p50_seconds': float(np.quantile(times, .5)),
                                     'p95_seconds': float(np.quantile(times, .95))}
                              for name, times in groups.items()}}
