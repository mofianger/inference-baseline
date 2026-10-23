"""Official OpenPI eager sampler; preprocessed GPU tensors to full actions."""
from pathlib import Path
from types import SimpleNamespace


class Pi05ReferenceAdapter:
    def __init__(self, config, protocol, samples):
        from safetensors.torch import load_model
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        self.protocol = protocol
        self.model = PI0Pytorch(Pi0Config(pi05=True, dtype="float32", pytorch_compile_mode=None))
        load_model(self.model, str(Path(config["checkpoint"]) / "model.safetensors"), strict=True)
        self.model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        self.model.to(config["device"]).eval()
        self.metadata = {"implementation": "official OpenPI eager sampler",
                         "precision": "official selected BF16 with FP32 stem, norms, projections and Euler state",
                         "boundary": "prepared GPU images/tokens/masks/noise through full 10-step sampler",
                         "preprocessing": "already normalized 224x224; upstream evaluation preprocessing performs only views and Python packaging"}

    def infer(self, sample):
        t = sample["inputs"]
        keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        observation = SimpleNamespace(
            images={k: t["pixel_values"][:, i] for i, k in enumerate(keys)},
            image_masks={k: t["image_mask"][:, i] for i, k in enumerate(keys)},
            state=t["state"], tokenized_prompt=t["input_ids"],
            tokenized_prompt_mask=t["attention_mask"], token_ar_mask=None, token_loss_mask=None)
        return self.model.sample_actions(sample["noise"].device, observation,
                                         noise=sample["noise"], num_steps=self.protocol.denoising_steps)
