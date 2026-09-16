"""Scoring of a reconstructed edit set against the typed-delta ground truth.

Pure functions, no framework imports, so they can be unit-tested and reused
outside a pipeline run.

Conventions (frozen, they match the thesis protocol):
- both sets empty            -> 1.0 on every metric (correct abstention)
- truth empty, prediction not -> 0.0 (the judge invented an edit)
- truth not empty, prediction empty -> 0.0 (nothing recovered)

Edits are matched strictly:
- edges as canonical pairs (min, max) for undirected graphs
- feature changes as (node, feature) pairs; the values are not compared,
  a narrative rarely carries them exactly
"""
import json
import re

EDIT_TYPES = ('edges_added', 'edges_removed', 'features_changed')


def calculate_metrics(set_true, set_pred):
    """Jaccard, precision, recall and F1 between two sets."""
    if not set_true and not set_pred:
        return {'jaccard': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}

    tp = len(set_true & set_pred)
    union = len(set_true | set_pred)
    precision = tp / len(set_pred) if set_pred else 0.0
    recall = tp / len(set_true) if set_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {'jaccard': tp / union if union else 0.0,
            'precision': precision, 'recall': recall, 'f1': f1}


def edge_key(edge, directed=False):
    u, v = int(edge[0]), int(edge[1])
    return (u, v) if directed else (min(u, v), max(u, v))


def feature_key(change):
    return (int(change['node']), str(change['feature']))


def delta_to_sets(delta, directed=False):
    """Turn a typed delta (ground truth or extraction) into one set per type."""
    sets = {}
    for edit_type in ('edges_added', 'edges_removed'):
        sets[edit_type] = {edge_key(e, directed) for e in delta.get(edit_type) or []}
    sets['features_changed'] = {feature_key(c) for c in delta.get('features_changed') or []}
    return sets


def score_delta(truth, prediction, directed=False):
    """Per-type metrics plus a structural aggregate over both edge types."""
    truth_sets = delta_to_sets(truth, directed)
    pred_sets = delta_to_sets(prediction, directed)

    scores = {edit_type: calculate_metrics(truth_sets[edit_type], pred_sets[edit_type])
              for edit_type in EDIT_TYPES}

    # Edges tagged by direction, so an added edge does not match a removed one
    truth_edges = {('+',) + e for e in truth_sets['edges_added']} | {('-',) + e for e in truth_sets['edges_removed']}
    pred_edges = {('+',) + e for e in pred_sets['edges_added']} | {('-',) + e for e in pred_sets['edges_removed']}
    scores['edges'] = calculate_metrics(truth_edges, pred_edges)
    return scores


def parse_extraction(text):
    """Parse the judge's answer into a typed delta.

    Accepts a bare JSON object or one inside a ```json fence. Edges may come
    as [u, v] lists or "u-v" / "u -- v" strings; feature changes as
    {node, feature} objects or "node:feature" strings. Returns None when no
    JSON object can be read.
    """
    if not text:
        return None
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    candidate = match.group(1) if match else text
    start, end = candidate.find('{'), candidate.rfind('}')
    if start < 0 or end < 0:
        return None
    try:
        raw = json.loads(candidate[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None

    delta = {'edges_added': [], 'edges_removed': [], 'features_changed': []}
    for edit_type in ('edges_added', 'edges_removed'):
        for item in raw.get(edit_type) or []:
            edge = _parse_edge(item)
            if edge is not None:
                delta[edit_type].append(edge)
    for item in raw.get('features_changed') or []:
        change = _parse_feature_change(item)
        if change is not None:
            delta['features_changed'].append(change)
    return delta


def _parse_edge(item):
    if isinstance(item, (list, tuple)) and len(item) == 2:
        try:
            return [int(item[0]), int(item[1])]
        except (TypeError, ValueError):
            return None
    if isinstance(item, str):
        found = re.findall(r'\d+', item)
        if len(found) == 2:
            return [int(found[0]), int(found[1])]
    return None


def _parse_feature_change(item):
    if isinstance(item, dict) and 'node' in item and 'feature' in item:
        try:
            return {'node': int(item['node']), 'feature': str(item['feature'])}
        except (TypeError, ValueError):
            return None
    if isinstance(item, str) and ':' in item:
        node, feature = item.split(':', 1)
        if node.strip().isdigit():
            return {'node': int(node), 'feature': feature.strip()}
    return None
