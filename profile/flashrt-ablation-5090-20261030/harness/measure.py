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
 # Warm-cache throughput cross-check: amortize event/graph boundary over 64 tasks.
 # Subtract a matched reset-only batch for mutating tasks, retaining both raw costs.
 count=64
 def reset_batch():
  for _ in range(count):reset()
 base=graph(reset_batch);batches={}
 for k,fn in funcs.items():
  def batch(fn=fn):
   for _ in range(count):reset();fn()
  batches[k]=graph(batch)
 for g in batches.values():
  for _ in range(5):g.replay()
 def elapsed(g):
  start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
  start.record();g.replay();end.record();end.synchronize();return start.elapsed_time(end)*1000/count
 raw={k:[] for k in batches};controls={k:[] for k in batches}
 for rnd in range(rounds):
  for k in (list(batches) if rnd%2==0 else list(batches)[::-1]):
   aa=[];bb=[]
   for _ in range(10):
    if rnd%2:aa.append(elapsed(batches[k]));bb.append(elapsed(base))
    else:bb.append(elapsed(base));aa.append(elapsed(batches[k]))
   raw[k].append(statistics.median(aa));controls[k].append(statistics.median(bb))
 for k in batches:
  v=out['warm'][k];v['event_window_median_us']=v['median_us'];v['event_window_round_median_us']=v['round_median_us']
  adjusted=[x-y for x,y in zip(raw[k],controls[k])]
  v.update(median_us=statistics.median(adjusted),round_median_us=adjusted,batched_including_reset_us=raw[k],batched_reset_only_us=controls[k],batch_count=count,method='64-task batch, subtract matched reset-only batch; estimate under warm-cache repetition')
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
