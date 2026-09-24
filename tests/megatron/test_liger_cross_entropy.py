# Copyright (c) ModelScope Contributors. All rights reserved.
"""`--cross_entropy_fusion_impl liger` against an fp32 reference.

`_patch_liger_cross_entropy` rebinds the function that `LanguageModule.compute_language_model_loss`
calls for 'native'. This checks the rebound function on Megatron's layout ([s, b, V] bf16 logits,
-100 for ignored labels) on one CUDA or NPU device::

    PYTHONPATH=. python tests/megatron/test_liger_cross_entropy.py
"""
import torch
import torch.nn.functional as F
import unittest

SEQ, BATCH, VOCAB = 1024, 2, 151936


class TestLigerCrossEntropy(unittest.TestCase):

    def test_matches_fp32_reference(self):
        from megatron.core.models.common.language_module import language_module

        from swift.megatron.init import _patch_liger_cross_entropy
        from swift.utils import get_device
        _patch_liger_cross_entropy()

        device = get_device()
        torch.manual_seed(0)
        logits = torch.randn(SEQ, BATCH, VOCAB, dtype=torch.bfloat16).to(device)
        labels = torch.randint(0, VOCAB, (SEQ, BATCH))
        labels[::3] = -100
        labels = labels.to(device)
        grad_output = torch.rand(SEQ, BATCH).to(device)

        ref_logits = logits.float().requires_grad_()
        ref_loss = F.cross_entropy(ref_logits.view(-1, VOCAB), labels.view(-1), reduction='none').view(SEQ, BATCH)
        ref_loss.backward(grad_output)

        # Like TE, liger may overwrite its input with the gradient, so it gets its own copy.
        liger_logits = logits.clone().requires_grad_()
        loss = language_module.fused_vocab_parallel_cross_entropy(liger_logits, labels)
        loss.backward(grad_output)

        self.assertEqual(loss.dtype, torch.float32)
        # liger stores the loss in the logits' dtype, so it carries bf16 rounding (relative error <= 2**-9).
        torch.testing.assert_close(loss, ref_loss.detach(), rtol=4e-3, atol=0)
        # The gradient is bf16 either way; compare against the reference rounded to bf16.
        torch.testing.assert_close(liger_logits.grad, ref_logits.grad.to(torch.bfloat16))


if __name__ == '__main__':
    unittest.main()
