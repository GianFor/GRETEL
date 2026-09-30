"""Scoring of a reconstructed edit set against the typed-delta ground truth.

Pure functions, no framework imports, so they can be unit-tested and reused
outside a pipeline run.

Conventions (frozen, they match the thesis protocol):
- both sets empty            -> 1.0 on every metric (correct abstention)
- truth empty, prediction not -> 0.0 (the judge invented an edit)
- truth not empty, prediction empty -> 0.0 (nothing recovered)

Edits are matched strictly:
- edges as canonical pairs (min, max) for undirected graphs
- feature_match=identity: (node, feature), the thesis's legacy convention
- feature_match=transition: (node, feature, from, to), exact numeric values
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


def feature_key(change, feature_match='identity'):
    key = (int(change['node']), str(change['feature']))
    return key + (change['from'], change['to']) if feature_match == 'transition' else key


def delta_to_sets(delta, directed=False, feature_match='identity'):
    """Turn a typed delta (ground truth or extraction) into one set per type."""
    sets = {}
    for edit_type in ('edges_added', 'edges_removed'):
        sets[edit_type] = {edge_key(e, directed) for e in delta.get(edit_type) or []}
    sets['features_changed'] = {feature_key(c, feature_match) for c in delta.get('features_changed') or []}
    for key in ('nodes_added', 'nodes_removed'):
        sets[key] = set(delta.get(key) or [])
    return sets


def score_delta(truth, prediction, directed=False, feature_match='identity'):
    """Per-type metrics plus a structural aggregate over both edge types."""
    if feature_match not in ('identity', 'transition'):
        raise ValueError('feature_match must be identity or transition')
    truth_sets = delta_to_sets(truth, directed, feature_match)
    pred_sets = delta_to_sets(prediction, directed, feature_match)

    scores = {edit_type: calculate_metrics(truth_sets[edit_type], pred_sets[edit_type])
              for edit_type in EDIT_TYPES}

    # Edges tagged by direction, so an added edge does not match a removed one
    truth_edges = {('+',) + e for e in truth_sets['edges_added']} | {('-',) + e for e in truth_sets['edges_removed']}
    pred_edges = {('+',) + e for e in pred_sets['edges_added']} | {('-',) + e for e in pred_sets['edges_removed']}
    scores['edges'] = calculate_metrics(truth_edges, pred_edges)
    for key in ('nodes_added', 'nodes_removed'):
        scores[key] = calculate_metrics(truth_sets[key], pred_sets[key])
    return scores


def parse_extraction(text, strict=False, directed=False, feature_match='identity'):
    """Parse the judge's answer into a typed delta.

    Accepts a bare JSON object or one inside a ```json fence. Edges may come
    as [u, v] lists or "u-v" / "u -- v" strings; feature changes as
    {node, feature} objects or "node:feature" strings. Returns None when no
    JSON object can be read.
    """
    if strict:
        from src.utils.probe_common import read_object, validate_delta
        try:
            return validate_delta(read_object(text), directed, feature_match=feature_match)
        except ValueError:
            return None
    if not isinstance(text, str) or not text:
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
    if not isinstance(raw, dict) or not any(k in raw for k in EDIT_TYPES):
        return None

    delta = {'edges_added': [], 'edges_removed': [], 'features_changed': []}
    for edit_type in ('edges_added', 'edges_removed'):
        values = raw.get(edit_type, [])
        if not isinstance(values, list):
            return None
        for item in values:
            edge = _parse_edge(item)
            if edge is None:
                return None
            delta[edit_type].append(edge)
    values = raw.get('features_changed', [])
    if not isinstance(values, list):
        return None
    for item in values:
        change = _parse_feature_change(item)
        if change is None:
            return None
        delta['features_changed'].append(change)
    return delta


def _parse_edge(item):
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return list(item) if all(type(v) is int and v >= 0 for v in item) else None
    if isinstance(item, str):
        match = re.fullmatch(r'\s*(\d+)\s*(?:--|->|-)\s*(\d+)\s*', item)
        if match:
            return [int(match[1]), int(match[2])]
    return None


def _parse_feature_change(item):
    if isinstance(item, dict) and 'node' in item and 'feature' in item:
        if type(item['node']) is not int or item['node'] < 0 or not str(item['feature']).strip():
            return None
        return {'node': item['node'], 'feature': str(item['feature'])}
    if isinstance(item, str) and ':' in item:
        node, feature = item.split(':', 1)
        if node.strip().isdigit():
            return {'node': int(node), 'feature': feature.strip()}
    return None
