"""OpenPI pi05 base: three active views and discretized state tokens."""
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Pi05Protocol:
    name: str = "pi05-base-b1-v3-model-only-v1"
    model: str = "pi0.5"
    batch: int = 1
    views: int = 3
    denoising_steps: int = 10
    action_horizon: int = 50
    action_dim: int = 32
    state_dim: int = 32
    image_size: list = field(default_factory=lambda: [224, 224])
    max_token_len: int = 200
    input_dtype: str = "float32"
    boundary: str = "device_tensor_to_device_normalized_action"
    timer: str = "synchronized_wall"

    def to_dict(self):
        return asdict(self)

    def sensitivity_groups(self):
        return {"pixel_values": ("pixel_values",),
                "state_tokens": ("state", "input_ids", "attention_mask"), "noise": ("noise",)}

    def validate(self, sample):
        import torch
        if self.to_dict() != Pi05Protocol().to_dict():
            raise ValueError("Changed pi05 geometry/boundary requires a new protocol version")
        t = sample["inputs"]
        shapes = {"pixel_values": (1, 3, 3, 224, 224), "image_mask": (1, 3),
                  "state": (1, 32), "input_ids": (1, 200), "attention_mask": (1, 200)}
        if set(t) != set(shapes):
            raise ValueError("pi05 input keys differ from protocol")
        for key, shape in shapes.items():
            if tuple(t[key].shape) != shape:
                raise ValueError(f"Wrong pi05 {key} shape")
            dtype = torch.bool if key in ("image_mask", "attention_mask") else torch.int64 if key == "input_ids" else torch.float32
            if t[key].dtype != dtype:
                raise ValueError(f"Wrong pi05 {key} dtype")
        if not t["image_mask"].all().item():
            raise ValueError("All three views must be active")
        for key in ("pixel_values", "state"):
            if not torch.isfinite(t[key]).all().item() or t[key].abs().max().item() > 1:
                raise ValueError(f"pi05 {key} must be normalized to [-1, 1]")
        mask = t["attention_mask"]
        if not mask[:, 0].all().item() or ((~mask[:, :-1]) & mask[:, 1:]).any().item():
            raise ValueError("Expected nonempty right-padded text")
        if (t["input_ids"] < 0).any().item() or (t["input_ids"] >= 257152).any().item():
            raise ValueError("Token outside PaliGemma vocabulary")
        noise = sample["noise"]
        if tuple(noise.shape) != (1, 50, 32) or noise.dtype != torch.float32 or not torch.isfinite(noise).all().item():
            raise ValueError("Expected finite FP32 initial noise [1, 50, 32]")
