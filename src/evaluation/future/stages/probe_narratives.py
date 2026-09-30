"""Generate the paper's two-part answers using the upstream narrative task."""
import copy
import json

from src.core.factory_base import get_instance_kvargs
from src.evaluation.future.stages.stage import Stage
from src.LLMexplaneability.prompt_generator import SYSTEM_PROMPT
from src.utils import probe_graph
from src.utils.probe_common import OUTPUT_SCHEMA, call_many, generation_outcome, inverse_delta
from src.utils.typed_delta import typed_delta_from_instances
from src.utils.context import clean_cfg


class ProbeNarratives(Stage):

    def check_configuration(self):
        super().check_configuration()
        p = self.local_config['parameters']
        if 'generator' not in p:
            raise ValueError('ProbeNarratives requires a generator LLM snippet')
        p.setdefault('node_features', None)
        p.setdefault('atol', 0.0)
        p.setdefault('node_matching', 'position')
        p.setdefault('generator_family', None)
        if p['node_matching'] != 'position':
            raise ValueError('A semantic node mapper has not been configured; use explicit positional matching')

    def init(self):
        super().init()
        # Upstream MainPipeline constructs stages twice. Delay costly loading
        # until this actual stage is used, so discarded stages reserve no GPU.
        self.generator = None

    def process(self, explanation):
        p = self.local_config['parameters']
        if self.generator is None:
            self.generator = get_instance_kvargs(p['generator']['class'], {'context': self.context, 'local_config': p['generator']})
        feature_map = explanation.dataset.node_features_map or {}
        if p['node_features'] is not None:
            if any(name not in feature_map for name in p['node_features']):
                raise ValueError('Unknown selected node features')
            columns = [feature_map[name] for name in p['node_features']]
        else:
            columns = None
        original = explanation.input_instance
        input_label = self._predict(explanation, original)
        factual_text = probe_graph.graph_text(original, input_label, feature_map)
        records, prompts, owners = [], [], []
        # Do not silently lose failed counterfactual generation from the denominator.
        for cf in explanation.counterfactual_instances or [None]:
            record = {'status': 'input_error', 'direct_output': None, 'inverse_output': None,
                      'direct_generation': {'status': 'not_attempted', 'call_status': 'not_attempted'},
                      'inverse_generation': {'status': 'not_attempted', 'call_status': 'not_attempted'}}
            records.append(record)
            if cf is None:
                record['error'] = 'No counterfactual produced'
                continue
            target = self._predict(explanation, cf)
            record.update(input=probe_graph.snapshot(original), counterfactual=probe_graph.snapshot(cf),
                          input_label=input_label, target_label=target, graph_text=factual_text,
                          inverse_graph_text=probe_graph.graph_text(cf, target, feature_map))
            if probe_graph.unsupported_changes(original, cf):
                record.update(status='unsupported_delta', error='Direction, graph features or existing edge values/attributes changed; TypedDelta does not describe these edits')
                continue
            if original.node_features.shape[1] != cf.node_features.shape[1]:
                record.update(status='unsupported_delta', error='Node-feature column sets differ between the two graphs')
                continue
            if target == input_label:
                record.update(status='not_counterfactual', error='The final graph does not flip the oracle')
                continue
            truth = typed_delta_from_instances(original, cf, feature_columns=columns,
                                              feature_names={v: k for k, v in feature_map.items()}, atol=p['atol'])
            record.update(status='success', truth=truth)
            # Give the generator the states and edits; never synthesize its answer afterwards.
            prompt_truth = {key: truth[key] for key in ('nodes_added', 'nodes_removed', 'edges_added', 'edges_removed', 'features_changed')}
            for direction, before, after, delta in (
                ('direct', factual_text, record['inverse_graph_text'], prompt_truth),
                ('inverse', record['inverse_graph_text'], factual_text, inverse_delta(prompt_truth))
            ):
                system = SYSTEM_PROMPT.split('OUTPUT FORMAT:')[0] + '\n' + OUTPUT_SCHEMA + '\n' + explanation.dataset.domain
                prompt = f'--- ORIGINAL ---\n{before}\n--- COUNTERFACTUAL ---\n{after}\n--- MODIFICATIONS ---\n{json.dumps(delta)}'
                prompts.append((system, prompt))
                owners.append((len(records) - 1, direction))
        for record in records:
            record['input_status'] = record['status']
        for (index, direction), answer in zip(owners, call_many(self.generator, prompts)):
            records[index][direction + '_output'] = answer['judge_output']
            records[index][direction + '_generation'] = generation_outcome(answer, original.directed)
        for record in records:
            if record['input_status'] == 'success':
                statuses = [record[d + '_generation']['status'] for d in ('direct', 'inverse')]
                record['status'] = 'success' if statuses == ['success', 'success'] else 'partial_error'
        self.write_into_explanation(explanation, {
            'schema_version': 1, 'node_matching': p['node_matching'], 'feature_map': feature_map,
            'feature_columns': columns, 'atol': p['atol'], 'domain': explanation.dataset.domain,
            'generator': copy.deepcopy(clean_cfg(self.generator.local_config)),
            'generator_family': p['generator_family'],
            'dataset': copy.deepcopy(clean_cfg(explanation.dataset.local_config)),
            'oracle': copy.deepcopy(clean_cfg(explanation.oracle.local_config)), 'counterfactuals': records,
        })
        return explanation

    @staticmethod
    def _predict(explanation, instance):
        try:
            return int(explanation.oracle.predict(instance))
        finally:
            explanation.oracle._call_counter -= 1
