# SPDX-License-Identifier: Apache-2.0
"""Experimental canonical CPU ownership and demand-only bounded execution."""
import os
import re

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.utils import set_weight_attrs


def enabled():
    value = os.environ.get('VLLM_CPU_BACKED_EXPERTS', '0')
    if value not in ('0', '1'):
        raise ValueError('VLLM_CPU_BACKED_EXPERTS must be 0 or 1')
    return value == '1'


def validate_speculation(spec, async_scheduling=False):
    if spec is None:
        return
    draft = spec.draft_model_config
    hf = getattr(draft.hf_config, 'model', draft.hf_config)
    if (async_scheduling or spec.method != 'eagle3' or spec.num_speculative_tokens != 2
            or spec.enable_adaptive_verification
            or spec.num_speculative_tokens_per_batch_size is not None
            or spec.rejection_sample_method != 'standard'
            or draft.quantization is not None or draft.dtype != torch.bfloat16
            or hf.model_type != 'llama'
            or draft.architectures != ['LlamaForCausalLMEagle3']
            or hf.hidden_size != 2048
            or hf.num_hidden_layers != 1):
        raise ValueError('CPU-backed speculation supports only synchronous fixed two-token BF16 dense Llama EAGLE-3')


def validate_config(config, quant_config, prefix):
    validate_speculation(config.speculative_config, config.scheduler_config.async_scheduling)
    model, parallel = config.model_config, config.parallel_config
    hf = model.hf_text_config
    match = re.search(r'(?:^|\.)layers\.(\d+)\.mlp\.experts$', prefix)
    if (config.device_config.device_type != 'cuda'
            or config.compilation_config.cudagraph_mode.name != 'NONE'
            or hf.model_type != 'qwen3_moe' or hf.hidden_size != 2048
            or hf.moe_intermediate_size != 768 or hf.num_experts != 128
            or hf.num_hidden_layers != 48 or hf.num_experts_per_tok != 8
            or hf.hidden_act != 'silu' or model.dtype != torch.bfloat16
            or quant_config is not None or model.quantization is not None
            or not model.enforce_eager or match is None
            or config.scheduler_config.max_num_seqs != 1
            or parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1 or parallel.enable_expert_parallel
            or parallel.enable_eplb or parallel.use_sequence_parallel_moe
            or config.lora_config is not None
            or int(config.compilation_config.mode) != 0
            or config.kernel_config.moe_backend not in ('auto', 'triton')
            or parallel.enable_dbo
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or config.offload_config.uva.cpu_offload_gb != 0
            or config.offload_config.prefetch.offload_group_size != 0
            or str(config.load_config.load_format) not in ('auto', 'safetensors')):
        raise ValueError('CPU-backed experts require canonical BF16 Qwen3-30B-A3B, '
                         'single GPU/request, eager, safetensors, no quantization, '
                         'parallelism, LoRA, or parameter offloading')
    return int(match[1])


class CPUBackedMoEMethod(FusedMoEMethodBase):
    requires_device_loading = False
    supports_pre_processed_weights = False

    def __init__(self, moe, layer_id):
        super().__init__(moe)
        self.layer_id = layer_id
        self.loaded = set()
        self.ready = False

    def maybe_roundup_sizes(self, hidden_size, intermediate_size_per_partition, *args):
        return hidden_size, intermediate_size_per_partition

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **attrs):
        if ((num_experts, hidden_size, intermediate_size_per_partition) != (128, 2048, 768)
                or params_dtype != torch.bfloat16 or self.moe.has_bias
                or not self.moe.is_act_and_mul or layer.is_fused_checkpoint_transposed):
            raise ValueError('Unsupported CPU-backed expert layout')
        original_loader = attrs['weight_loader']

        def tracked_loader(*args, **kwargs):
            result = original_loader(*args, **kwargs)
            if result is True:
                self.loaded.add((kwargs['expert_id'], kwargs['shard_id']))
            return result

        attrs = {**attrs, 'weight_loader': tracked_loader}
        for name, shape in [('w13_weight', (128, 1536, 2048)),
                            ('w2_weight', (128, 2048, 768))]:
            parameter = torch.nn.Parameter(torch.empty(shape, dtype=params_dtype,
                device='cpu', pin_memory=True), requires_grad=False)
            set_weight_attrs(parameter, attrs)
            parameter.weight_loader_numel = parameter.numel()
            layer.register_parameter(name, parameter)

    def process_weights_after_loading(self, layer):
        expected = {(e, shard) for e in range(128) for shard in ('w1', 'w2', 'w3')}
        if self.loaded != expected:
            raise RuntimeError(f'Incomplete expert checkpoint: {len(self.loaded)}/384 projections')
        if any(p.device.type != 'cpu' or not p.is_pinned()
               for p in (layer.w13_weight, layer.w2_weight)):
            raise RuntimeError('Expert ownership moved out of pinned CPU storage')
        self.ready = True

    def get_fused_moe_quant_config(self, layer):
        return None

    def apply(self, layer, x, topk_weights, topk_ids,
              shared_experts=None, shared_experts_input=None):
        if not self.ready or not hasattr(self, 'cached_execution'):
            raise RuntimeError('CPU-backed cache must be bound after loading')
        if shared_experts is not None or layer.apply_router_weight_on_input:
            raise ValueError('Unsupported shared experts or input-side routing weights')
        return self.cached_execution.apply(self.layer_id, x, topk_weights, topk_ids)


def expert_views(model):
    """Borrow immutable CPU views, retaining original layer/expert identities."""
    result = {}
    for module in model.modules():
        method = getattr(module, 'quant_method', None)
        if isinstance(method, CPUBackedMoEMethod):
            if not method.ready:
                raise RuntimeError('Post-load validation must complete before cache construction')
            for expert in range(128):
                key = (method.layer_id, expert)
                if key in result:
                    raise RuntimeError('Duplicate expert identity')
                result[key] = (module.w13_weight[expert].detach(), module.w2_weight[expert].detach())
    if not result:
        raise ValueError('Model has no CPU-backed experts')
    return result
