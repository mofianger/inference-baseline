import statistics,time,torch

def graph(fn):
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for _ in range(3):fn()
 torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=stream):fn()
 torch.cuda.synchronize();return g

def paired_graphs(funcs,reset=lambda:None,rounds=6,reps=40):
 """Time whole task with events; reset/eviction precedes start on the same stream."""
 evict=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
 out={}
 for cache in ['warm','l2_pressure_128MiB']:
  records={k:[] for k in funcs};gs={};events={}
  for k,fn in funcs.items():
   starts=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(reps)]
   ends=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(reps)]
   def batch(fn=fn,starts=starts,ends=ends):
    for j in range(reps):
     reset()
     if cache!='warm':evict.zero_()
     starts[j].record();fn();ends[j].record()
   gs[k]=graph(batch);events[k]=(starts,ends)
  for k,g in gs.items():
   for _ in range(5):g.replay()
  torch.cuda.synchronize()
  for rnd in range(rounds):
   for k in (list(gs) if rnd%2==0 else list(gs)[::-1]):
    gs[k].replay()
    torch.cuda.synchronize()
    starts,ends=events[k]
    records[k].append([s.elapsed_time(e)*1000 for s,e in zip(starts,ends)])
  out[cache]={k:dict(round_median_us=[statistics.median(x) for x in rr],median_us=statistics.median(v for x in rr for v in x),samples_us=rr) for k,rr in records.items()}
 return out

def wall_pairs(funcs,rounds=6,reps=100):
 result={k:[] for k in funcs}
 for rnd in range(rounds):
  for k in (list(funcs) if rnd%2==0 else list(funcs)[::-1]):
   fn=funcs[k];t=time.perf_counter()
   while time.perf_counter()-t<2:fn()
   arr=[]
   for _ in range(reps):
    torch.cuda.synchronize();t=time.perf_counter();fn();torch.cuda.synchronize();arr.append((time.perf_counter()-t)*1000)
   result[k].append(arr)
 return {k:dict(round_median_ms=[statistics.median(x) for x in rr],median_ms=statistics.median(v for x in rr for v in x),samples_ms=rr) for k,rr in result.items()}
