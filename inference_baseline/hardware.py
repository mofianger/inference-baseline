import platform
import time
from .io import command


class TorchDevice:
    """Device adapter: CUDA now; CPU for contract tests, other accelerators extensible."""
    def __init__(self, device="cuda:0"):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type not in ("cuda", "cpu"):
            raise ValueError("Register a synchronization backend for this device first")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

    def sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def time(self, fn):
        self.sync()
        start = time.perf_counter_ns()
        output = fn()
        self.sync()
        return output, (time.perf_counter_ns() - start) / 1e6

    def metadata(self):
        t = self.torch
        result = {"platform": platform.platform(), "python": platform.python_version(),
                  "torch": t.__version__, "torch_cuda": t.version.cuda, "device": str(self.device),
                  "cpu": platform.processor(), "cpu_threads": t.get_num_threads(),
                  "interop_threads": t.get_num_interop_threads(),
                  "tf32_matmul": t.backends.cuda.matmul.allow_tf32,
                  "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,power.limit,pstate,clocks.sm,clocks.mem,temperature.gpu", "--format=csv,noheader"])}
        if self.device.type == "cuda":
            p = t.cuda.get_device_properties(self.device)
            result.update(name=p.name, capability=[p.major, p.minor], total_memory=p.total_memory)
        return result
