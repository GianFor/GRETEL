from src.future.explanation.base import Explanation
from src.evaluation.future.stages.stage import Stage
from src.utils.typed_delta import typed_delta_from_instances


class TypedDelta(Stage):
    """Records the typed edit set between the input graph and each of its
    counterfactuals, with the oracle prediction on both sides.

    This is the ground truth the narrative probes are scored against. When the
    explainer is a generate-minimize pipeline, the generator's (seed)
    counterfactuals are recorded too, so the same instance can be compared
    before and after minimization.

    Parameters:
    - node_features: names of the node features to compare (default: all).
      Leave out features derived from the topology by a manipulator.
    - atol: tolerance below which a feature value counts as unchanged.
    """

    def check_configuration(self):
        super().check_configuration()
        self.logger = self.context.logger
        self.local_config['parameters'].setdefault('node_features', None)
        self.local_config['parameters'].setdefault('atol', 0.0)

    def init(self):
        super().init()
        self.node_features = self.local_config['parameters']['node_features']
        self.atol = self.local_config['parameters']['atol']

    def process(self, explanation: Explanation) -> Explanation:
        input_inst = explanation.input_instance
        input_lbl = self._predict(explanation, input_inst)

        # {name: column} as built by the dataset and its manipulators
        features_map = explanation.dataset.node_features_map or {}
        feature_names = {col: name for name, col in features_map.items()}
        feature_columns = None
        if self.node_features is not None:
            missing = [name for name in self.node_features if name not in features_map]
            if missing:
                raise ValueError(f'Unknown node features {missing}, available: {list(features_map)}')
            feature_columns = [features_map[name] for name in self.node_features]

        value = {
            'input_label': input_lbl,
            'counterfactuals': [self._record(explanation, input_inst, input_lbl, cf, feature_columns, feature_names)
                                for cf in explanation.counterfactual_instances],
        }

        generator_explanation = explanation.info.get('generator_explanation')
        if generator_explanation is not None:
            value['generator_counterfactuals'] = [
                self._record(explanation, input_inst, input_lbl, cf, feature_columns, feature_names)
                for cf in generator_explanation.counterfactual_instances]

        self.write_into_explanation(explanation, value)
        return explanation

    def _record(self, explanation, input_inst, input_lbl, cf, feature_columns, feature_names):
        cf_lbl = self._predict(explanation, cf)
        record = {'counterfactual_label': cf_lbl, 'is_counterfactual': cf_lbl != input_lbl}
        record.update(typed_delta_from_instances(input_inst, cf,
                                                 feature_columns=feature_columns,
                                                 feature_names=feature_names,
                                                 atol=self.atol))
        return record

    @staticmethod
    def _predict(explanation, instance):
        # Evaluation calls must not count towards the explainer's oracle calls
        label = explanation.oracle.predict(instance)
        explanation.oracle._call_counter -= 1
        return int(label)
