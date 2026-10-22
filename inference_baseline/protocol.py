from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelProtocol:
    name: str
    model: str
    batch: int
    views: int
    denoising_steps: int
    action_horizon: int
    action_dim: int
    state_dim: int
    state_history: int
    image_size: list
    embodiment_id: int
    input_dtype: str = "bfloat16"
    boundary: str = "device_tensor_to_device_normalized_action"
    timer: str = "synchronized_wall"

    @classmethod
    def from_checkpoint(cls, config, embodiment_id):
        if config.get("model_type") != "Gr00tN1d7":
            raise ValueError("This model plugin requires Gr00tN1d7")
        return cls("groot-n17-b1-v3-model-only-v1", "GR00T N1.7", 1, 3,
                   config["num_inference_timesteps"], config["action_horizon"],
                   config["max_action_dim"], config["max_state_dim"],
                   config.get("state_history_length", 1), config["image_target_size"],
                   embodiment_id)

    def to_dict(self):
        return asdict(self)

    def validate(self, sample):
        t = sample["inputs"]
        required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "state", "embodiment_id"}
        if set(t) != required:
            raise ValueError(f"Input keys differ: {set(t) ^ required}")
        if tuple(t["state"].shape) != (self.batch, self.state_history, self.state_dim):
            raise ValueError("State shape does not match protocol")
        if tuple(sample["noise"].shape) != (self.batch, self.action_horizon, self.action_dim):
            raise ValueError("Noise/action shape does not match protocol")
        if tuple(t["image_grid_thw"].shape) != (self.views, 3):
            raise ValueError("Expected exactly three image views")
        if t["input_ids"].shape[0] != self.batch or t["input_ids"].shape != t["attention_mask"].shape:
            raise ValueError("Token batch/mask mismatch")
        if t["embodiment_id"].tolist() != [self.embodiment_id]:
            raise ValueError("Embodiment mismatch")
        if self.boundary != "device_tensor_to_device_normalized_action" or self.timer != "synchronized_wall":
            raise ValueError("Runner v1 cannot silently execute a different timing boundary")
        for key in ("pixel_values", "state"):
            if str(t[key].dtype) != "torch." + self.input_dtype:
                raise ValueError(f"{key} dtype differs from frozen protocol")
        if str(sample["noise"].dtype) != "torch." + self.input_dtype:
            raise ValueError("Initial noise dtype differs from protocol")
        for key in ("input_ids", "attention_mask", "image_grid_thw", "embodiment_id"):
            if str(t[key].dtype) != "torch.int64":
                raise ValueError(f"Expected int64 metadata for {key}")
        grid = t["image_grid_thw"]
        if (grid[:, 0] != 1).any().item():
            raise ValueError("This protocol requires single-frame image views, not video")
        expected_grid = [self.image_size[0] // 16, self.image_size[1] // 16]
        if grid[:, 1:].tolist() != [expected_grid] * self.views:
            raise ValueError("Image grid differs from official checkpoint image size")
        rows = int(t["image_grid_thw"].prod(dim=1).sum().item())
        if t["pixel_values"].shape[0] != rows:
            raise ValueError("Patch count does not match image grid")
