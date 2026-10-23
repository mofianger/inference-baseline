"""Freeze synthetic pi05 model inputs with official state tokenization."""
from pathlib import Path
from .io import checkpoint_manifest, revision, sha256, write_json
from .pi05_protocol import Pi05Protocol


def prepare(config, destination, count=5):
    import numpy as np
    import sentencepiece
    import torch
    from openpi.models.tokenizer import PaligemmaTokenizer
    from .adapters.pi05_reference import Pi05ReferenceAdapter
    from .dataset import move_sample
    if count < 3:
        raise ValueError("Need calibration and at least two evaluation samples")
    root = Path(destination)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Refusing to overwrite frozen fixtures")
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.get("cpu_threads", 8))
    torch.set_num_interop_threads(1)
    protocol = Pi05Protocol()
    # Use the upstream method with an explicitly pinned local tokenizer asset.
    tokenizer = object.__new__(PaligemmaTokenizer)
    tokenizer._max_len = protocol.max_token_len
    tokenizer._tokenizer = sentencepiece.SentencePieceProcessor(model_file=config["tokenizer"])
    prompt = "Pick up the object and place it on the table."
    samples = []
    for i in range(count):
        rng = np.random.default_rng(20261023 + i)
        pixels = rng.integers(0, 256, (1, 3, 3, 224, 224), dtype=np.uint8).astype(np.float32) / 127.5 - 1
        state = rng.uniform(-0.8, 0.8, (1, 32)).astype(np.float32)
        ids, mask = tokenizer.tokenize(prompt, state[0])
        inputs = {"pixel_values": torch.from_numpy(pixels), "image_mask": torch.ones((1, 3), dtype=torch.bool),
                  "state": torch.from_numpy(state), "input_ids": torch.from_numpy(ids.astype(np.int64))[None],
                  "attention_mask": torch.from_numpy(mask.astype(bool))[None]}
        sample = {"id": f"synthetic-{i:04d}", "role": "calibration" if i == 0 else "evaluation",
                  "inputs": inputs, "noise": torch.from_numpy(rng.standard_normal((1, 50, 32)).astype(np.float32))}
        protocol.validate(sample)
        samples.append(sample)
    adapter = Pi05ReferenceAdapter(config, protocol, samples)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = {"schema": 1, "protocol": protocol.to_dict(), "checkpoint": checkpoint_manifest(config["checkpoint"]),
                "tokenizer_sha256": sha256(config["tokenizer"]), "prompt": prompt,
                "data_kind": "synthetic_three_distinct_views_discretized_state", "samples": [],
                "reference": {**adapter.metadata, "source": revision(config["engine_source"]), "torch": torch.__version__}}
    with torch.inference_mode():
        for sample in samples:
            output = adapter.infer(move_sample(sample, config["device"]))
            sample["reference_actions"] = output.float().cpu().clone()
            if tuple(output.shape) != (1, 50, 32) or not torch.isfinite(output).all().item():
                raise ValueError("Invalid official reference output")
            filename = sample["id"] + ".pt"
            torch.save(sample, root / filename)
            manifest["samples"].append({"id": sample["id"], "role": sample["role"], "file": filename,
                "sha256": sha256(root / filename),
                "inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in sample["inputs"].items()}})
            print(f"prepared {sample['id']} action={tuple(output.shape)}", flush=True)
    write_json(root / "manifest.json", manifest)
    return manifest
