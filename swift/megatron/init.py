# Copyright (c) ModelScope Contributors. All rights reserved.
import concurrent.futures
import inspect
import logging
import os
import sys
import torch
import torch.distributed as dist
from contextlib import contextmanager
from copy import copy, deepcopy
from tqdm import tqdm
from transformers.modeling_utils import custom_object_save
from transformers.utils import is_torch_npu_available
from typing import Union

from swift.model import get_model_processor, save_checkpoint
from swift.utils import (HfConfigFactory, disable_safe_ddp_context_use_barrier, get_logger, get_modules_to_not_convert,
                         get_multimodal_target_regex, is_master, split_list)

logger = get_logger()


def _patch__batched_p2p_ops():
    from megatron.core.pipeline_parallel import p2p_communication

    _batched_p2p_ops_origin = p2p_communication._batched_p2p_ops

    def _batched_p2p_ops(**kwargs):
        kwargs['group'] = None
        return _batched_p2p_ops_origin(**kwargs)

    p2p_communication._batched_p2p_ops = _batched_p2p_ops


def _patch_torch_FileSystemReader():
    from torch.distributed.checkpoint.filesystem import FileSystemReader
    from torch.futures import Future
    if getattr(FileSystemReader.read_data, '_swift_patched', False):
        return
    _origin_read_data = FileSystemReader.read_data
    _origin__slice_file = FileSystemReader._slice_file
    READER_MAX_WORKERS = int(os.environ.get('MCORE_READER_MAX_WORKERS', '16'))

    @contextmanager
    def _patch__slice_file(prog_bar):

        def _slice_file(self, *args, **kwargs):
            prog_bar.update()
            return _origin__slice_file(self, *args, **kwargs)

        FileSystemReader._slice_file = _slice_file
        try:
            yield
        finally:
            FileSystemReader._slice_file = _origin__slice_file

    def read_data(self, plan, planner):

        def _worker(plan_shard):
            _origin_read_data(self, plan_shard, planner)

        prog_bar = tqdm(total=len(plan.items), dynamic_ncols=True, desc='Loading: ')
        try:
            plan_shards = split_list(plan.items, READER_MAX_WORKERS, contiguous=False)
            with _patch__slice_file(prog_bar):
                with concurrent.futures.ThreadPoolExecutor(max_workers=READER_MAX_WORKERS) as pool:
                    futures = []
                    for i in range(READER_MAX_WORKERS):
                        plan_shard = copy(plan)
                        plan_shard.items = plan_shards[i]
                        futures.append(pool.submit(_worker, plan_shard))
                    concurrent.futures.wait(futures)
                    for future in futures:
                        future.result()
        finally:
            prog_bar.close()
        fut: Future = Future()
        fut.set_result(None)
        return fut

    read_data._swift_patched = True
    FileSystemReader.read_data = read_data


def _dcp_validation_returns_errors(default_planner) -> bool:
    """Whether `_validate_global_plan` is expected to return a list of error messages."""
    try:
        # The caller is what defines the contract, so it is the most reliable thing to inspect.
        source = inspect.getsource(default_planner.DefaultSavePlanner._create_global_plan)
        return 'validation_errors' in source
    except (OSError, TypeError):
        pass
    annotation = inspect.signature(default_planner._validate_global_plan).return_annotation
    if annotation is inspect.Signature.empty:
        logger.warning(f'Could not determine the `_validate_global_plan` contract of torch=={torch.__version__}; '
                       'assuming the legacy boolean form.')
        return False
    return annotation not in (bool, 'bool')


def _patch_validate_non_overlapping_shards_metadata():
    # too slow
    from torch.distributed._shard.sharded_tensor import api
    from torch.distributed._shard.sharding_spec import api as api2
    from torch.distributed.checkpoint import default_planner

    def validate_non_overlapping_shards_metadata(*args, **kwargs):
        pass

    api.validate_non_overlapping_shards_metadata = validate_non_overlapping_shards_metadata
    api2.validate_non_overlapping_shards_metadata = validate_non_overlapping_shards_metadata

    # The return contract changed across torch versions: it used to be a bool (falsy meaning
    # "invalid"), while newer versions return a list of error messages (empty meaning "valid").
    # Returning the wrong type is not harmless -- a bool sends the newer caller into its error
    # branch, where `'; '.join(True)` raises `TypeError: can only join an iterable` and buries the
    # real reason for the failure.
    if _dcp_validation_returns_errors(default_planner):

        def _validate_global_plan(*args, **kwargs):
            return []
    else:

        def _validate_global_plan(*args, **kwargs):
            return True

    default_planner._validate_global_plan = _validate_global_plan


def _patch_unified_memory():
    if is_torch_npu_available():
        return

    from torch.utils import cpp_extension
    load_inline = cpp_extension.load_inline

    def _new_load_inline(*args, **kwargs):
        name = kwargs.get('name')
        if name == 'managed_alloc_runtime':
            raise RuntimeError
        return load_inline(*args, **kwargs)

    # not create unified memory mempool
    cpp_extension.load_inline = _new_load_inline
    try:
        from megatron.core.inference import unified_memory
    except Exception:
        pass
    finally:
        cpp_extension.load_inline = load_inline


def _patch_vllm_qwen4_exp_config():
    """Backfill config defaults vLLM's qwen4_exp config class does not declare.

    vLLM ships its own `Qwen4ExpTextConfig` and registers it for the
    `qwen4_exp_text` model type via `AutoConfig.register(..., exist_ok=True)`,
    which replaces the Transformers class in the process-wide `CONFIG_MAPPING`.
    Under colocate GRPO the rollout engine lives in the training process, so every
    later `AutoConfig.from_pretrained` resolves to vLLM's class -- including the
    one used to build the dummy HF model when saving. Transformers' own
    `Qwen4ExpTextNGramEmbedding` then reads `config.seed`, which vLLM's class does
    not define, and released checkpoints do not carry it either, so saving dies
    with `AttributeError: 'Qwen4ExpTextConfig' object has no attribute 'seed'`.

    Only class-level defaults are added, and only for names vLLM is missing, so an
    explicit value from `config.json` still wins (instance `__dict__` takes
    precedence) and a future vLLM that declares them is left untouched.
    """
    if 'vllm' not in sys.modules:
        return  # vLLM never loaded -> the Transformers class is still in charge
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        # Imported by module path on purpose: AutoConfig lookups already resolve to
        # vLLM's class at this point, so they cannot supply the reference defaults.
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig, Qwen4ExpTextConfig
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
    except Exception:
        return  # no qwen4_exp on either side -> nothing to mirror
    # vLLM only registers the outer model type it actually loaded; the text config
    # class is reached through that class's `sub_configs`, never via CONFIG_MAPPING.
    # So gate on the outer override being live, then fix up both classes.
    active_outer = CONFIG_MAPPING._extra_content.get('qwen4_exp') if hasattr(CONFIG_MAPPING, '_extra_content') else None
    if active_outer is None or active_outer is Qwen4ExpConfig:
        return  # Transformers' class still in charge -> nothing to do
    for model_type, hf_cls in (('qwen4_exp', Qwen4ExpConfig), ('qwen4_exp_text', Qwen4ExpTextConfig)):
        try:
            vllm_cls = _CONFIG_REGISTRY[model_type]  # LazyConfigDict resolves on access
        except Exception:
            continue
        if vllm_cls is hf_cls:
            continue
        for name in ('seed', ):
            if not hasattr(vllm_cls, name) and hasattr(hf_cls, name):
                setattr(vllm_cls, name, getattr(hf_cls, name))
                logger.info(f'Backfilled `{name}` default onto vLLM {model_type} config '
                            f'(vLLM does not declare it; needed by the Transformers modeling code).')


def _patch_vllm_glm5_next_config():
    """Backfill the config aliases vLLM's glm5_next config classes do not declare.

    Sibling of `_patch_vllm_qwen4_exp_config` and the same failure class: vLLM ships its own
    `Glm5NextTextConfig` and registers it for the `glm5_next` / `glm5_next_text` model types via
    `AutoConfig.register(..., exist_ok=True)`, which replaces the Transformers classes in the
    process-wide `CONFIG_MAPPING`. Under colocate GRPO the rollout engine lives in the training
    process, so every later `AutoConfig.from_pretrained` resolves to vLLM's classes -- including
    the one used to build the dummy HF model when saving. Transformers' own `Glm5NextTextExperts`
    reads `config.num_local_experts`, which resolves only through
    `attribute_map = {'num_local_experts': 'n_routed_experts'}`; vLLM's class keeps the real field
    but declares no map, so saving dies with
    `AttributeError: 'Glm5NextTextConfig' object has no attribute 'num_local_experts'`.

    `attribute_map` is merged key by key instead of probed with `hasattr`, because
    `PretrainedConfig` already defines it as `{}` -- the attribute exists, the entries do not.
    Only class-level entries are added, so an explicit value from `config.json` still wins
    (instance `__dict__` takes precedence) and a future vLLM that declares the map is untouched.
    """
    if 'vllm' not in sys.modules:
        return  # vLLM never loaded -> the Transformers classes are still in charge
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        # Imported by module path on purpose: AutoConfig lookups already resolve to vLLM's
        # classes at this point, so they cannot supply the reference aliases.
        from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig, Glm5NextTextConfig
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
    except Exception:
        return  # no glm5_next on either side -> nothing to mirror
    # vLLM only registers the outer model type it actually loaded; the text config class is
    # reached through that class's `sub_configs`, never via CONFIG_MAPPING. So gate on the outer
    # override being live, then fix up both classes.
    active_outer = CONFIG_MAPPING._extra_content.get('glm5_next') if hasattr(CONFIG_MAPPING, '_extra_content') else None
    if active_outer is None or active_outer is Glm5NextConfig:
        return  # Transformers' class still in charge -> nothing to do
    for model_type, hf_cls in (('glm5_next', Glm5NextConfig), ('glm5_next_text', Glm5NextTextConfig)):
        try:
            vllm_cls = _CONFIG_REGISTRY[model_type]  # LazyConfigDict resolves on access
        except Exception:
            continue
        if vllm_cls is hf_cls:
            continue
        hf_map = getattr(hf_cls, 'attribute_map', None) or {}
        vllm_map = getattr(vllm_cls, 'attribute_map', None) or {}
        missing = {alias: target for alias, target in hf_map.items() if alias not in vllm_map}
        if missing:
            setattr(vllm_cls, 'attribute_map', {**vllm_map, **missing})
            logger.info(f'Backfilled `attribute_map` {missing} onto vLLM {model_type} config '
                        f'(vLLM does not declare it; needed by the Transformers modeling code).')


def _patch_mcore_bridge():
    import mcore_bridge
    from mcore_bridge import GPTBridge
    logger.info(f'mcore_bridge.__version__: {mcore_bridge.__version__}')
    origin_save_weights = GPTBridge.save_weights
    origin_parameters = inspect.signature(origin_save_weights).parameters

    def save_weights(
        self,
        mg_models,
        output_dir: str,
        peft_format: bool = False,
        max_shard_size: str = '5GB',
        args=None,
        processor=None,
        save_missing_weights: Union[bool, str] = False,
    ) -> None:
        kwargs = {}
        if 'save_missing_weights' in origin_parameters:
            kwargs['save_missing_weights'] = save_missing_weights
        elif save_missing_weights:
            logger.warning('The installed `mcore-bridge` does not support `save_missing_weights`. '
                           'Please upgrade it via `pip install mcore-bridge -U`. Ignoring this parameter.')
        origin_save_weights(
            self, mg_models, output_dir, peft_format=peft_format, max_shard_size=max_shard_size, **kwargs)
        if processor is None or args is None:
            return
        hf_config = self.config.hf_config
        hf_config = deepcopy(hf_config)
        if is_master() and not hasattr(self, 'hf_model'):
            if hasattr(self, 'get_hf_meta_model'):
                self.hf_model = self.get_hf_meta_model()
                self.hf_model.model_meta = processor.model_meta
                self.hf_model.model_info = processor.model_info
            else:
                _patch_vllm_qwen4_exp_config()
                _patch_vllm_glm5_next_config()
                with torch.device('meta'), disable_safe_ddp_context_use_barrier():
                    self.hf_model = get_model_processor(
                        args.model_dir, model_type=args.model_type, return_dummy_model=True)[0]

        if is_master():
            if peft_format:
                peft_config = copy(mg_models[0].peft_config[self._adapter_name])
                if self.config.task_type == 'seq_cls':
                    peft_config.task_type = 'SEQ_CLS'
                if self.is_multimodal and 'all-linear' in args.target_modules:
                    peft_config.target_modules = get_multimodal_target_regex(
                        self.hf_model,
                        freeze_llm=args.freeze_llm,
                        freeze_vit=args.freeze_vit,
                        freeze_aligner=args.freeze_aligner,
                        include_embedding='all-embedding' in args.target_modules,
                        exclude_router='all-router' not in args.target_modules)
                else:
                    assert not isinstance(peft_config.target_modules, str), (
                        'target_regex is not currently supported for LoRA conversion. Please set `--merge_lora true`.')
                    peft_config.target_modules = self._peft_target_modules
                peft_config.modules_to_save = self._peft_modules_to_save
                peft_config.save_pretrained(output_dir)
            else:
                config = self.config
                llm_config = HfConfigFactory.get_text_config(hf_config)
                if config.mtp_num_layers:
                    for key in ['num_nextn_predict_layers', 'mtp_num_hidden_layers']:
                        if hasattr(llm_config, key):
                            setattr(llm_config, key, config.mtp_num_layers)
                            break
                    else:
                        llm_config.num_nextn_predict_layers = config.mtp_num_layers
                HfConfigFactory.del_config_attr(hf_config, 'quantization_config')
                expert_dtype = None
                if config.fp8 is not None and config.fp8_recipe == 'blockwise' and config.fp8_param:
                    from transformers.utils.quantization_config import FineGrainedFP8Config
                    modules_to_not_convert = get_modules_to_not_convert(self.hf_model)
                    if hasattr(self, '_fp8_skip_modules'):
                        modules_to_not_convert = (modules_to_not_convert or []) + list(self._fp8_skip_modules)
                    hf_config.quantization_config = FineGrainedFP8Config(modules_to_not_convert=modules_to_not_convert)
                    expert_dtype = 'fp8'
                if args.model_type == 'deepseek_v4':
                    HfConfigFactory.set_config_attr(hf_config, 'expert_dtype', expert_dtype)
                hf_config.save_pretrained(output_dir)
                if getattr(self.hf_model, '_auto_class') is not None:
                    try:
                        custom_object_save(self.hf_model, output_dir, config=hf_config)
                    except FileNotFoundError as e:
                        logger.error(f'custom_object_save Error: {e}')
                save_checkpoint(
                    None,
                    processor,
                    output_dir,
                    model_dirs=[args.model_dir],
                    additional_saved_files=self.hf_model.model_meta.additional_saved_files)
            logger.info(f'Successfully saved `safetensors` model weights in `{output_dir}`.')
        dist.barrier()  # Ensure all weights are saved completely

    GPTBridge.save_weights = save_weights


def _patch_liger_cross_entropy():
    """Serve `cross_entropy_fusion_impl='liger'` on Ascend NPU through Megatron's 'native' call site.

    `LanguageModule` imports `fused_vocab_parallel_cross_entropy` by name, so it has to be rebound in
    `language_module` itself; patching `megatron.core.fusions` would not reach the caller.

    With TP>1 this uses liger's vocab-parallel kernel. With TP=1 the local logits cover the full vocabulary,
    and liger's Ascend 2-D kernel is several times faster.

    The kernels are picked here instead of by liger. MindSpeed maps `torch.cuda` to `torch_npu` before liger
    is imported, so liger's `infer_device()` returns 'cuda': it keeps its CUDA kernels and a block size of
    32768, which the Ascend compiler rejects.

    The 2-D kernel computes the loss in fp32, but liger's `cross_entropy_forward` stores it in a buffer of the
    logits' dtype, so bf16 logits would give a bf16-rounded loss. The kernel is launched here instead, with an
    fp32 loss buffer and the launch config liger uses for bf16 logits, so the loss matches Megatron's native
    implementation and liger's vocab-parallel kernel. The backward is liger's.
    """
    import liger_kernel.ops.vocab_parallel_cross_entropy as liger_vocab_parallel
    from liger_kernel.megatron import LigerMegatronCrossEntropy
    from liger_kernel.ops.backends._ascend.ops import cross_entropy as liger_ce
    from megatron.core.models.common.language_module import language_module
    liger_vocab_parallel.MAX_FUSED_SIZE = 2048  # liger's own value for 'npu'
    vocab_parallel_cross_entropy = LigerMegatronCrossEntropy()

    class LigerCrossEntropy(torch.autograd.Function):

        @staticmethod
        def forward(ctx, logits, target):
            n_rows, v = logits.shape
            loss = torch.zeros(n_rows, dtype=torch.float32, device=logits.device)  # ignored rows are not written
            lse = torch.empty_like(loss)
            # [inv_n, inv_sum_weight, weight_sum]; reduction='none' without weight reads only inv_n
            ce_stats = torch.ones(3, dtype=torch.float32, device=logits.device)
            liger_ce.liger_cross_entropy_forward_kernel[(min(liger_ce.get_npu_core_count(), n_rows), )](
                X_ptr=logits,
                X_stride=logits.stride(-2),
                Y_ptr=target,
                weight_ptr=None,
                loss_ptr=loss,
                z_loss_ptr=None,
                lse_ptr=lse,
                token_accuracy_ptr=None,
                token_accuracy_stride=0,
                predicted_tokens_ptr=None,
                predicted_tokens_stride=0,
                n_cols=v,
                n_rows=n_rows,
                ce_stats_ptr=ce_stats,
                ignore_index=-100,
                ls_eps=0.0,
                lse_square_scale=0.0,
                label_smoothing=0.0,
                reduction='none',
                softcap=None,
                RETURN_Z_LOSS=False,
                RETURN_LSE=True,
                RETURN_TOKEN_ACCURACY=False,
                RETURN_PREDICTED_TOKENS=False,
                BLOCK_SIZE=liger_ce.get_optimal_block_size(v, has_gradients=False),
                HAS_WEIGHT=False,
                HAS_SOFTCAPPING=False,
            )
            ctx.save_for_backward(logits, target, lse, ce_stats)
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            logits, target, lse, ce_stats = ctx.saved_tensors
            # (input, target, weight, lse, grad_output, ignore_index, lse_square_scale, label_smoothing, reduction,
            #  softcap, ce_stats)
            grad_input = liger_ce.cross_entropy_backward(logits, target, None, lse, grad_output, -100, 0.0, 0.0, 'none',
                                                         None, ce_stats)
            return grad_input, None

    def liger_vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group=None):
        if tp_group is not None and tp_group.size() > 1:
            return vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group=tp_group)
        s, b, v = vocab_parallel_logits.shape
        return LigerCrossEntropy.apply(vocab_parallel_logits.view(-1, v), target.view(-1)).view(s, b)

    language_module.fused_vocab_parallel_cross_entropy = liger_vocab_parallel_cross_entropy


def init_megatron_env():
    os.environ.pop('VLLM_USE_MODELSCOPE', None)
    logging_level = logging.root.level
    _patch_unified_memory()
    if is_torch_npu_available():
        from swift.model.npu_patcher import patch_mindspeed_fla_gdn_implementation
        patch_mindspeed_fla_gdn_implementation()
    _patch__batched_p2p_ops()
    logging.root.setLevel(logging_level)  # revert logger level
    try:
        _patch_torch_FileSystemReader()
    except Exception:
        logger.warning('Failed to patch FileSystemReader.')
    try:
        _patch_validate_non_overlapping_shards_metadata()
    except Exception:
        logger.warning('Patch validate_non_overlapping_shards_metadata failed.')
        pass
    import megatron.core
    logger.info(f'megatron.core.__version__: {megatron.core.__version__}')
