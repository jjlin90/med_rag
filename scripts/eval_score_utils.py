"""Score validation and complete-pair aggregation, without model/API dependencies."""
import json
import math
from numbers import Real
import re
import statistics


def valid_score(value):
    return (isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)


def parse_judge_score(raw):
    """Reject malformed JSON and invalid values; never clamp or invent a zero."""
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', (raw or '').strip(),
                     flags=re.IGNORECASE)
    data = json.loads(cleaned)
    value = data.get('score') if isinstance(data, dict) else None
    if not valid_score(value):
        raise ValueError('Judge score must be a finite number in [0, 1]')
    return float(value)


def make_pair_row(question, orig_f, orig_ar, grounded_f, grounded_ar):
    values = [float(v) if valid_score(v) else None
              for v in (orig_f, orig_ar, grounded_f, grounded_ar)]
    row = dict(zip(('oF', 'oAR', 'gF', 'gAR'), values))
    row.update(q=question, failed=any(v is None for v in values))
    row['dF'] = None if row['failed'] else row['gF'] - row['oF']
    row['dAR'] = None if row['failed'] else row['gAR'] - row['oAR']
    return row


def summarize_pairs(rows):
    """Use exactly the same complete question pairs for every reported mean."""
    valid = [r for r in rows if all(valid_score(r.get(k))
                                   for k in ('oF', 'oAR', 'gF', 'gAR'))]
    def mean(key):
        return round(statistics.mean(r[key] for r in valid), 4) if valid else None
    summary = {'orig': {'F': mean('oF'), 'AR': mean('oAR')},
               'grnd': {'F': mean('gF'), 'AR': mean('gAR')}}
    summary['delta'] = {
        metric: round(statistics.mean(r[g] - r[o] for r in valid), 4) if valid else None
        for metric, o, g in (('F', 'oF', 'gF'), ('AR', 'oAR', 'gAR'))}
    return summary, len(valid)


def validate_frozen_contexts(originals, grounded):
    """An answer-only comparison must use the same evidence on both sides."""
    source = {row['question']: row for row in originals}
    for row in grounded:
        original = source.get(row['question'])
        if (original is not None and 'contexts' in row
                and row['contexts'] != original.get('contexts', [])):
            raise ValueError('Grounded contexts differ from the original; use full re-evaluation')


def require_independent_judge(judge, *datasets):
    generators = {row.get('_gen_model') for dataset in datasets for row in dataset
                  if row.get('_gen_model')}
    if judge in generators:
        raise ValueError('Judge matches a generation model; choose an independent judge')
