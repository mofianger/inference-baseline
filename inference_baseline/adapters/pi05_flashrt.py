"""FlashRT pi05 GPU-only boundary including dynamic token embeddings."""
import ctypes
import hashlib
from pathlib import Path


class Pi05FlashRTAdapter:
    def __init__(self, config, protocol, samples):
        import numpy as np
        import torch
        from flash_rt.frontends.torch import pi05_rtx as upstream
        self.torch = torch
        self.protocol = protocol
        options = config.get("options", {})
        precision = options.get("prefix_precision", "fp8")
        seed = options.get("calibration_seed", 0)
        torch.manual_seed(seed)
        # The model-only boundary has no robot-space normalization. Do not
        # invent normalization stats just to satisfy the raw-observation API.
        class PreparedFrontend(upstream.Pi05TorchFrontendRtx):
            def _load_norm_stats(self, checkpoint_dir):
                self.norm_stats = None

            def _stack_images(self, sample):
                return sample["inputs"]["pixel_values"][0].permute(0, 2, 3, 1).to(torch.bfloat16).contiguous()

        reader = upstream._StateReader
        class CanonicalReader(reader):
            def get_tensor(self, key):
                if key == "paligemma_with_expert.paligemma.lm_head.weight" and key not in self.keys():
                    key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                return super().get_tensor(key)
        # Construction is serialized within this one-process benchmark. Restore
        # the module symbol immediately; no upstream checkout edits are needed.
        upstream._StateReader = CanonicalReader
        try:
            self.frontend = PreparedFrontend(config["checkpoint"], num_views=3, chunk_size=50,
                max_prompt_len=200, num_steps=10, cache_frames=1, state_prompt_mode="fixed",
                use_cuda_graph=True, use_fp8=options.get("use_fp8", True), prefix_precision=precision,
                decoder_precision=options.get("decoder_precision", "fp8"))
        finally:
            upstream._StateReader = reader
        f = self.frontend
        f._set_prompt_fixed(200)
        f.attn_backend.set_fixed_shape(True)
        f.pipeline.set_language_embeds(np.zeros((200, 2048), dtype=np.uint16))
        self.rope = torch.from_numpy(f.pipeline._rope_table_np.view(np.uint16).copy()).view(torch.bfloat16).cuda()
        self.action_positions = torch.arange(50, device="cuda")
        calibration = [s for s in samples if s["role"] == "calibration"]
        if len(calibration) != 1:
            raise ValueError("FlashRT pi05 recipe requires exactly one frozen calibration sample")
        self.update_prompt(calibration[0], torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        f._last_prompt_len = int(calibration[0]["inputs"]["attention_mask"].sum().item())
        f.calibrate(calibration)
        self.metadata = {"implementation": "FlashRT Pi05Pipeline full graph + timed GPU token embeddings",
                         "prefix_precision": precision, "decoder_precision": options.get("decoder_precision", "fp8"),
                         "use_fp8": options.get("use_fp8", True), "cache_frames": 1,
                         "calibration_sample": calibration[0]["id"], "calibration_seed": seed,
                         "weight_alias": "PaliGemma lm_head -> identical input embedding",
                         "normalization": "excluded; normalized model actions returned directly"}
        from flash_rt import flash_rt_kernels, flash_rt_fa2
        self.metadata["kernel_binaries"] = {
            m.__name__: {"path": m.__file__, "sha256": hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()}
            for m in (flash_rt_kernels, flash_rt_fa2)}

    def update_prompt(self, sample, stream):
        torch, f = self.torch, self.frontend
        t = sample["inputs"]
        embeds = torch.nn.functional.embedding(t["input_ids"][0], f.embedding_weight)
        embeds = (embeds * float(embeds.shape[-1] ** 0.5)) * t["attention_mask"][0, :, None]
        f._copy_tensor_to_pipeline_buf_stream(embeds.contiguous(), f.pipeline._lang_embeds_buf, stream)
        valid = t["attention_mask"].sum(dim=1) + f.pipeline.vision_seq_enc
        f.attn_backend.enc_seqused.copy_(valid)
        f.attn_backend.dec_seqused.copy_(valid + self.protocol.action_horizon)
        f.attn_backend.dec_devpos.copy_(valid)
        rope = self.rope.index_select(0, self.action_positions + valid[0])
        f._copy_tensor_to_pipeline_buf_stream(rope, f.pipeline.bufs["decoder_rope_weights"], stream)

    def infer(self, sample):
        torch, f = self.torch, self.frontend
        with torch.cuda.stream(f._graph_torch_stream):
            stream = f._graph_torch_stream.cuda_stream
            self.update_prompt(sample, stream)
            f._img_buf.copy_(sample["inputs"]["pixel_values"][0].permute(0, 2, 3, 1))
            f._noise_buf.copy_(sample["noise"][0])
            f._copy_tensor_to_pipeline_buf_stream(f._img_buf, f.pipeline.input_images_buf, stream)
            f._copy_tensor_to_pipeline_buf_stream(f._noise_buf, f.pipeline.input_noise_buf, stream)
            pointer = f.pipeline.forward(stream=stream)
            rc = f._cudart.cudaMemcpyAsync(ctypes.c_void_p(f._noise_out.data_ptr()), ctypes.c_void_p(pointer),
                                          f._noise_out.numel() * f._noise_out.element_size(), 3, stream)
            if rc:
                raise RuntimeError(f"FlashRT output D2D copy failed: {rc}")
        torch.cuda.current_stream().wait_stream(f._graph_torch_stream)
        return f._noise_out.unsqueeze(0)

    def runtime_stats(self):
        f = self.frontend
        if f.pipeline._graph is None:
            raise RuntimeError("FlashRT pi05 full graph was not captured")
        return {"full_graph": True, "steps": f._num_steps, "fixed_prompt_capacity": f.pipeline.max_prompt_len,
                "prefix_precision": f._prefix_precision, "decoder_precision": f._decoder_precision,
                "use_fp8": f.pipeline.use_fp8, "use_fp8_decoder": f.pipeline.use_fp8_decoder,
                "nvfp4_decoder_active": bool(f.pipeline._dec_fp4),
                "nvfp4_prefix_ffn_active": bool(f.pipeline._ffn_fp4),
                "force_bf16": f._force_bf16}
