"""Official model, with instance-local deterministic initial-noise injection."""
import ast
import inspect
import textwrap
import types


def inject_noise(head):
    original = head.get_action_with_features.__func__
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    class ReplaceNoise(ast.NodeTransformer):
        count = 0
        def visit_Call(self, node):
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch" and node.func.attr == "randn"):
                self.count += 1
                return ast.copy_location(ast.parse("self._benchmark_initial_noise.clone()", mode="eval").body, node)
            return self.generic_visit(node)
    edit = ReplaceNoise()
    tree = edit.visit(tree)
    if edit.count != 1:
        raise RuntimeError("Official sampler changed: expected exactly one randn, refusing patch")
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, inspect.getsourcefile(original), "exec"), namespace)
    head.get_action_with_features = types.MethodType(namespace[original.__name__], head)


class ReferenceAdapter:
    def __init__(self, config, protocol, samples):
        import torch
        import gr00t.model
        from transformers import AutoConfig, AutoModel
        c = AutoConfig.from_pretrained(config["checkpoint"], local_files_only=True)
        # Keep the official registered model-name dispatch; HF_HOME resolves the cached backbone.
        c.use_flash_attention = False
        model, info = AutoModel.from_pretrained(config["checkpoint"], config=c,
            torch_dtype=torch.bfloat16, local_files_only=True, output_loading_info=True,
            transformers_loading_kwargs={"local_files_only": True, "trust_remote_code": False})
        if info["missing_keys"] or info["unexpected_keys"] or info.get("mismatched_keys"):
            raise RuntimeError(f"Checkpoint did not load exactly: {info}")
        self.model = model.to(config["device"]).eval()
        self.protocol = protocol
        inject_noise(self.model.action_head)
        self.metadata = {"implementation": "official eager SDPA", "precision": "BF16",
                         "noise_injection": "instance-local AST replacement of sole sampler randn; clone inside timing"}

    def infer(self, sample):
        self.model.action_head._benchmark_initial_noise = sample["noise"]
        return self.model.get_action(dict(sample["inputs"]))["action_pred"]
