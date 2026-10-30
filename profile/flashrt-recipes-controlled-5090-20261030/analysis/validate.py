import json,math,hashlib
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];R=root/'reports'
data=json.loads((root/'analysis/execution-data.json').read_text(encoding='utf-8'))
rr={r['id']:r for r in data['recipes']};out={}
for typ in ['fp8','nvfp4']:
 fused,unfused=(rr[typ+'-'+x] for x in ['fused','unfused'])
 a=torch.load(R/fused['id']/'wall-0/output.pt',map_location='cpu',weights_only=True).float().flatten()
 b=torch.load(R/unfused['id']/'wall-0/output.pt',map_location='cpu',weights_only=True).float().flatten()
 assert torch.isfinite(a).all() and torch.isfinite(b).all()
 error=dict(mae=(a-b).abs().mean().item(),rmse=(a-b).square().mean().sqrt().item(),max_abs=(a-b).abs().max().item(),cosine=torch.nn.functional.cosine_similarity(a,b,dim=0).item())
 def seq(r):
  return [(data['names'][e['n']],e['grid'],e['block']) for e in r['events'] if e['stage'].startswith('Step')]
 equal=seq(fused)==seq(unfused)
 scale='fp8-dit.json' if typ=='fp8' else 'nvfp4-cross.json'
 assert fused['calibration_hashes'][scale]==unfused['calibration_hashes'][scale]
 out[typ]=dict(saved_ms=unfused['wall_ms']-fused['wall_ms'],reduction_pct=100*(1-fused['wall_ms']/unfused['wall_ms']),paired_output_error=error,dit_kernel_sequence_equal=equal,dit_kernel_counts=[len(seq(r)) for r in [fused,unfused]],scale_hash=fused['calibration_hashes'][scale])
 assert equal,(typ,'DiT sequence mismatch')
for r in data['recipes']:
 assert len(r['rounds_ms'])==3 and len(r['all_spans'])==10
 assert all(e['d']>0 and e['s']>=0 for e in r['events'])
out['raw_sha256']={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*') if p.is_file() and (p.name in ['metadata.json','telemetry.csv','operations.json'] or p.name in ['trace.sqlite','mapping-v2.sqlite'])}
(root/'analysis/validation.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='raw_sha256'},indent=2))
