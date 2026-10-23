"""Deterministic processed-input fixture, without a simulator or task claims."""
from pathlib import Path
from .io import checkpoint_manifest, read_json, sha256, write_json, revision
from .protocol import ModelProtocol


def prepare(config, destination, count=5):
    if config.get("model") == "pi0.5":
        from .pi05_dataset import prepare as prepare_pi05
        return prepare_pi05(config, destination, count)
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from .adapters.reference import ReferenceAdapter
    torch.set_num_threads(config.get("cpu_threads", 8))
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if count < 3:
        raise ValueError("Need one calibration and at least two evaluation samples")
    root = Path(destination)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Dataset destination must be empty; do not overwrite frozen fixtures")
    root.mkdir(parents=True, exist_ok=True)
    ckpt = Path(config["checkpoint"])
    tag = "oxe_droid_relative_eef_relative_joint_3view"
    p = ModelProtocol.from_checkpoint(read_json(ckpt / "config.json"), read_json(ckpt / "embodiment_id.json")[tag])
    processor = AutoProcessor.from_pretrained(config["cosmos"], local_files_only=True)
    prompt = "Pick up the object and place it on the table."
    content = [{"type": "image"} for _ in range(p.views)] + [{"type": "text", "text": prompt}]
    text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    samples = []
    for i in range(count):
        rng = np.random.default_rng(20261023 + i)
        images = [Image.fromarray(rng.integers(0, 256, (*p.image_size, 3), dtype=np.uint8)) for _ in range(p.views)]
        inputs = dict(processor(text=[text], images=images, return_tensors="pt"))
        inputs = {k: v for k, v in inputs.items() if k in {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}}
        inputs["state"] = torch.from_numpy(rng.normal(0, 0.1, (1, p.state_history, p.state_dim)).astype(np.float32))
        inputs["embodiment_id"] = torch.tensor([p.embodiment_id], dtype=torch.int64)
        inputs = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in inputs.items()}
        noise = torch.from_numpy(rng.standard_normal((1, p.action_horizon, p.action_dim)).astype(np.float32)).bfloat16()
        sample = {"id": f"synthetic-{i:04d}", "inputs": inputs, "noise": noise,
                  "role": "calibration" if i == 0 else "evaluation"}
        p.validate(sample)
        samples.append(sample)
    config = dict(config)
    adapter = ReferenceAdapter(config, p, [])
    manifest = {"schema": 1, "protocol": p.to_dict(), "checkpoint": checkpoint_manifest(ckpt),
                "cosmos_assets": {f.name: sha256(f) for f in Path(config["cosmos"]).glob("*.json")},
                "data_kind": "synthetic_three_distinct_views", "prompt": prompt,
                "embodiment_tag": tag, "reference": {**adapter.metadata, "source": revision(config["engine_source"]),
                    "torch": torch.__version__, "transformers": __import__("transformers").__version__}, "samples": []}
    with torch.inference_mode():
        for sample in samples:
            gpu = move_sample(sample, config["device"])
            output = adapter.infer(gpu)
            sample["reference_actions"] = output.float().cpu().clone()
            # Shape-only/positional constants for FlashRT. No observation-dependent learned features are stored.
            qwen = adapter.model.backbone.model.model
            t = gpu["inputs"]
            pos, _ = qwen.get_rope_index(t["input_ids"], t["image_grid_thw"], attention_mask=t["attention_mask"])
            lm = qwen.language_model
            embeds = lm.embed_tokens(t["input_ids"])
            cos, sin = lm.rotary_emb(embeds, pos)
            sample["flashrt_constants"] = {
                "position_ids": pos.cpu(), "rope_cos": cos.cpu(), "rope_sin": sin.cpu(),
                "vision_pos_embed": qwen.visual.fast_pos_embed_interpolate(t["image_grid_thw"]).cpu(),
                "visual_pos_masks": (t["input_ids"] == qwen.config.image_token_id).cpu()}
            fn = f"{sample['id']}.pt"
            torch.save(sample, root / fn)
            manifest["samples"].append({"id": sample["id"], "file": fn, "role": sample["role"], "sha256": sha256(root / fn),
                "inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in sample["inputs"].items()}})
            print(f"prepared {sample['id']} action={tuple(output.shape)}", flush=True)
    write_json(root / "manifest.json", manifest)
    return manifest


def move_sample(sample, device):
    import torch
    def move(v):
        if isinstance(v, torch.Tensor):
            return v.to(device)
        if isinstance(v, dict):
            return {k: move(x) for k, x in v.items()}
        return v
    return move(sample)


def load_dataset(root):
    import torch
    root = Path(root)
    m = read_json(root / "manifest.json")
    if m["protocol"]["model"] == "pi0.5":
        from .pi05_protocol import Pi05Protocol
        p = Pi05Protocol(**m["protocol"])
    else:
        p = ModelProtocol(**m["protocol"])
    samples = []
    for item in m["samples"]:
        path = (root / item["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != item["sha256"]:
            raise ValueError(f"Invalid fixture path/hash: {item['file']}")
        sample = torch.load(path, weights_only=True, map_location="cpu")
        p.validate(sample)
        if sample["id"] != item["id"] or sample["role"] != item["role"]:
            raise ValueError("Manifest sample identity mismatch")
        samples.append(sample)
    return m, p, samples
