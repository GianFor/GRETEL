"""RECONSTRUCTION probe as a pipeline stage: can an independent judge recover
the edit set from the narrative alone?

Runs after LLMexplanation (which produces the narratives) and TypedDelta
(which records the ground truth). The judge is a component of this stage,
declared in its parameters, so it is never the generator held in
context.llm and its configuration is part of the stage's identity.

Loading generator and judge in one process needs room for both models. When
that does not fit, run the generation pipeline alone (lab/config/snippets/
generation_pipeline.json) and score the per-instance dumps afterwards with
scripts/run_probes.py, which uses the same code (src/utils/reconstruction_probe.py).

Parameters:
- judge: an LLM snippet, e.g.
    {"class": "src.LLMexplaneability.huggingface.HuggingFaceLLM",
     "parameters": {"model": "openai/gpt-oss-20b", "engine": "vllm"}}
- context: "off" (default) gives the judge only the generator output, the
  user-facing condition; "on" prepends the factual graph description
- mode: "full" (default) passes the raw generator output; "dict" passes only
  the narrative field of the two-part output schema. Outputs without that
  field are recorded as unparsed and not scored, so parse rates stay visible.
"""
from src.core.factory_base import get_instance_kvargs
from src.evaluation.future.stages.stage import Stage
from src.future.explanation.base import Explanation
from src.utils import reconstruction_probe

NARRATIVES_STAGE = 'src.evaluation.future.stages.llm_explanation.LLMexplanation'
DELTA_STAGE = 'src.evaluation.future.stages.typed_delta.TypedDelta'


class Reconstruction(Stage):

    def check_configuration(self):
        super().check_configuration()
        self.logger = self.context.logger
        p = self.local_config['parameters']
        if 'judge' not in p:
            raise ValueError('Reconstruction needs a "judge" LLM snippet in its parameters')
        p.setdefault('context', 'off')
        p.setdefault('mode', 'full')
        if p['context'] not in ('on', 'off'):
            raise ValueError('Reconstruction "context" must be "on" or "off"')
        if p['mode'] not in ('full', 'dict'):
            raise ValueError('Reconstruction "mode" must be "full" or "dict"')

    def init(self):
        super().init()
        p = self.local_config['parameters']
        self.use_context = p['context'] == 'on'
        self.mode = p['mode']
        self.judge = get_instance_kvargs(p['judge']['class'],
                                         {'context': self.context, 'local_config': p['judge']})

    def process(self, explanation: Explanation) -> Explanation:
        narratives = explanation.stages_info.get(NARRATIVES_STAGE)
        deltas = explanation.stages_info.get(DELTA_STAGE)
        if narratives is None or deltas is None:
            raise RuntimeError('Reconstruction must run after LLMexplanation and TypedDelta in the pipeline')

        directed = explanation.input_instance.directed
        items = [(output, graph_text, truth, directed)
                 for output, graph_text, truth in zip(narratives['direct_explanation'],
                                                      narratives['graph_text'],
                                                      deltas['counterfactuals'])]
        records = reconstruction_probe.run(self.judge, items, self.use_context, self.mode)
        for record in records:
            if record['status'] == 'judge_unparsed':
                self.logger.warning('Reconstruction: judge output could not be parsed')

        self.write_into_explanation(explanation, {
            'context': 'on' if self.use_context else 'off',
            'mode': self.mode,
            'counterfactuals': records,
        })
        return explanation
