"""Local HuggingFace backend for the LLM contract (system, prompt) -> text.

One class for generators and judges alike; the model is a config parameter,
so every run's identity carries the exact model, quantization and sampling
settings. Two engines behind the same interface:

- "transformers": plain HF generate, one prompt at a time. Needs no extra
  dependency.
- "vllm": batched inference through vLLM, an order of magnitude faster on a
  whole dataset. Needs `pip install vllm`.

Parameters (all hashed into the component name):
- model: HF repo id, e.g. "Qwen/Qwen3-8B"
- engine: "transformers" | "vllm"
- dtype: "bfloat16" | "float16" | "auto"
- quantization: None, or for vllm "awq" | "gptq" | "fp8"; for transformers
  "4bit" | "8bit" (bitsandbytes). MXFP4 checkpoints (gpt-oss) load as-is.
- temperature, top_p, seed, max_new_tokens: sampling controls
- enable_thinking: chat-template switch for hybrid models (Qwen3); False
  keeps the output to the answer only
- reasoning_effort: chat-template switch for gpt-oss ("low"|"medium"|"high")
- tensor_parallel_size, gpu_memory_utilization, max_model_len: vllm only

Weights are cached under HF_HOME; point it at a large disk.
"""
import re

from src.core.llm_base import LLM
from src.utils.logger import GLogger

# Hybrid models emit an (often empty) think block before the answer
_THINK_BLOCK = re.compile(r'<think>.*?</think>\s*', re.DOTALL)


class HuggingFaceLLM(LLM):

    def check_configuration(self):
        super().check_configuration()
        p = self.local_config['parameters']
        if 'model' not in p:
            raise ValueError('HuggingFaceLLM needs a "model" parameter (HF repo id)')
        p.setdefault('engine', 'transformers')
        if p['engine'] not in ('transformers', 'vllm'):
            raise ValueError('HuggingFaceLLM "engine" must be "transformers" or "vllm"')
        p.setdefault('dtype', 'bfloat16')
        p.setdefault('quantization', None)
        p.setdefault('temperature', 0.0)
        p.setdefault('top_p', 1.0)
        p.setdefault('seed', 0)
        p.setdefault('max_new_tokens', 1024)
        p.setdefault('enable_thinking', False)
        p.setdefault('reasoning_effort', None)
        p.setdefault('tensor_parallel_size', 1)
        p.setdefault('gpu_memory_utilization', 0.9)
        p.setdefault('max_model_len', None)

    def init(self):
        super().init()
        p = self.local_config['parameters']
        self.model_id = p['model']
        self.engine = p['engine']
        self.temperature = p['temperature']
        self.top_p = p['top_p']
        self.seed = p['seed']
        self.max_new_tokens = p['max_new_tokens']
        # Extra variables handed to the chat template; templates that do not
        # know a variable simply ignore it
        self.chat_kwargs = {'enable_thinking': bool(p['enable_thinking'])}
        if p['reasoning_effort'] is not None:
            self.chat_kwargs['reasoning_effort'] = p['reasoning_effort']

        if self.engine == 'vllm':
            self._init_vllm(p)
        else:
            self._init_transformers(p)
        GLogger.getLogger().info(f'LLM - {self.model_id} loaded with {self.engine}')

    # ------------------------------------------------------------------

    def _init_vllm(self, p):
        from vllm import LLM as VllmEngine, SamplingParams
        self.llm = VllmEngine(model=self.model_id,
                              dtype=p['dtype'],
                              quantization=p['quantization'],
                              tensor_parallel_size=p['tensor_parallel_size'],
                              gpu_memory_utilization=p['gpu_memory_utilization'],
                              max_model_len=p['max_model_len'],
                              seed=self.seed)
        self.sampling = SamplingParams(temperature=self.temperature, top_p=self.top_p,
                                       max_tokens=self.max_new_tokens, seed=self.seed)

    def _init_transformers(self, p):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {'device_map': 'auto'}
        if p['quantization'] in ('4bit', '8bit'):
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=p['quantization'] == '4bit',
                load_in_8bit=p['quantization'] == '8bit',
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type='nf4')
        elif p['dtype'] != 'auto':
            kwargs['dtype'] = getattr(torch, p['dtype'])
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self.model.eval()

    # ------------------------------------------------------------------

    @staticmethod
    def _messages(system, prompt):
        return [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]

    @staticmethod
    def _clean(text):
        return _THINK_BLOCK.sub('', text or '').strip()

    def explain_counterfactual(self, system, prompt):
        return self.explain_many([(system, prompt)])[0]

    def explain_many(self, pairs):
        """Answer a list of (system, prompt) pairs; batched under vllm."""
        messages = [self._messages(s, p) for s, p in pairs]
        if self.engine == 'vllm':
            outputs = self.llm.chat(messages, self.sampling, use_tqdm=False,
                                    chat_template_kwargs=self.chat_kwargs)
            return [self._clean(o.outputs[0].text) for o in outputs]
        return [self._generate_one(m) for m in messages]

    def _generate_one(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, **self.chat_kwargs)
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        self.torch.manual_seed(self.seed)
        generate_kwargs = {'max_new_tokens': self.max_new_tokens,
                           'pad_token_id': self.tokenizer.pad_token_id,
                           'do_sample': self.temperature > 0}
        if self.temperature > 0:
            generate_kwargs.update(temperature=self.temperature, top_p=self.top_p)
        with self.torch.no_grad():
            output = self.model.generate(**inputs, **generate_kwargs)
        generated = output[0][inputs['input_ids'].shape[1]:]
        return self._clean(self.tokenizer.decode(generated, skip_special_tokens=True))
