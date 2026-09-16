#!/usr/bin/env python3
"""Score the RECONSTRUCTION probe offline, over per-instance dumps.

Second pass of the two-pass protocol. The first pass runs the generation
pipeline (lab/config/snippets/generation_pipeline.json), which writes one
cf_<id>.json per instance with the typed delta and the narratives. This
script loads one judge, reads those dumps and scores every requested
condition, so generator and judge never share a GPU and a new judge never
requires regenerating narratives.

    python scripts/run_probes.py --config lab/config/probes/judge_gpt-oss-20b.jsonc \
                                 --results lab/output/results/probes-pilot

The probe config declares the judge and the conditions:

    {
      "experiment": {"scope": "probes"},
      "compose_strs": "./lab/config/snippets/default_store_paths.json",
      "judge_name": "gpt-oss-20b",
      "judge": {"class": "src.LLMexplaneability.huggingface.HuggingFaceLLM",
                "parameters": {"model": "openai/gpt-oss-20b", "engine": "vllm"}},
      "conditions": [{"context": "off", "mode": "full"}, {"context": "on", "mode": "full"}]
    }

Output, next to each dump: probes/<judge_name>/<ctx>_<mode>/cf_<id>.json, and
probes/<judge_name>/summary.csv with mean scores per condition and edit type.
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.factory_base import get_instance_kvargs  # noqa: E402
from src.utils import reconstruction_probe  # noqa: E402
from src.utils.context import Context  # noqa: E402

DELTA_KEY = 'TypedDelta'
NARRATIVES_KEY = 'LLMexplanation'
SCORE_TYPES = ('edges', 'edges_added', 'edges_removed', 'features_changed')


def load_dumps(results_root):
    """Every cf_<id>.json under results_root that carries both stages."""
    dumps = []
    for path in sorted(glob.glob(os.path.join(results_root, '**', 'cf_per_instance', '**', 'cf_*.json'),
                                 recursive=True)):
        with open(path) as f:
            payload = json.load(f)
        metrics = payload.get('metrics') or {}
        if DELTA_KEY in metrics and NARRATIVES_KEY in metrics:
            dumps.append((path, payload))
    return dumps


def items_of(payload):
    metrics = payload['metrics']
    narratives, deltas = metrics[NARRATIVES_KEY], metrics[DELTA_KEY]
    directed = bool(payload['input'].get('directed', False))
    return [(output, graph_text, truth, directed)
            for output, graph_text, truth in zip(narratives['direct_explanation'],
                                                 narratives['graph_text'],
                                                 deltas['counterfactuals'])]


def summarize(records):
    """Mean of every metric per edit type over successful records, plus counts."""
    row = {'n': len(records),
           'n_success': sum(r['status'] == 'success' for r in records),
           'n_unparsed': sum(r['status'] == 'unparsed' for r in records),
           'n_judge_unparsed': sum(r['status'] == 'judge_unparsed' for r in records)}
    scored = [r['scores'] for r in records if r['status'] == 'success']
    for edit_type in SCORE_TYPES:
        for metric in ('f1', 'precision', 'recall', 'jaccard'):
            values = [s[edit_type][metric] for s in scored]
            row[f'{edit_type}_{metric}'] = sum(values) / len(values) if values else None
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='probe config (judge + conditions)')
    parser.add_argument('--results', required=True, help='results root to scan for cf_per_instance dumps')
    parser.add_argument('--limit', type=int, default=None, help='score at most this many instances')
    args = parser.parse_args()

    context = Context.get_context(args.config)
    conf = context.conf
    judge_name = conf.get('judge_name') or conf['judge']['parameters']['model'].replace('/', '_')
    conditions = conf.get('conditions') or [{'context': 'off', 'mode': 'full'}]

    dumps = load_dumps(args.results)
    if args.limit:
        dumps = dumps[:args.limit]
    if not dumps:
        print(f'No dumps with {DELTA_KEY} and {NARRATIVES_KEY} under {args.results}')
        return 1
    print(f'{len(dumps)} instances, {len(conditions)} conditions, judge {judge_name}')

    judge = get_instance_kvargs(conf['judge']['class'], {'context': context, 'local_config': conf['judge']})

    summary_rows = []
    for condition in conditions:
        use_context = condition.get('context', 'off') == 'on'
        mode = condition.get('mode', 'full')
        tag = f"{'ctx-on' if use_context else 'ctx-off'}_mode-{mode}"

        # One batch for the whole condition, then split back per instance
        items, owners = [], []
        for index, (_, payload) in enumerate(dumps):
            for item in items_of(payload):
                items.append(item)
                owners.append(index)
        records = reconstruction_probe.run(judge, items, use_context, mode)

        per_instance = [[] for _ in dumps]
        for owner, record in zip(owners, records):
            per_instance[owner].append(record)

        for (path, payload), instance_records in zip(dumps, per_instance):
            out_dir = os.path.join(os.path.dirname(os.path.dirname(path)), 'probes', judge_name, tag)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, os.path.basename(path)), 'w') as f:
                json.dump({'id': payload['id'], 'judge': judge_name, 'context': condition.get('context', 'off'),
                           'mode': mode, 'counterfactuals': instance_records}, f, indent=1)

        row = {'judge': judge_name, 'condition': tag}
        row.update(summarize(records))
        summary_rows.append(row)
        print(f"{tag}: {row['n_success']}/{row['n']} scored, edges F1 = {row['edges_f1']}")

    summary_path = os.path.join(args.results, 'probes', f'summary_{judge_name}.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f'Summary written to {summary_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
