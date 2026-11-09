"""Measured NumPy streaming copies. No interpolation of partial copy operations."""
import threading,time
import numpy as np
class CopyWorkers:
 def __init__(self,n,array_bytes=256_000_000):
  self.n=n; self.array_bytes=array_bytes; self.stop=threading.Event(); self.go=threading.Event(); self.events=[[] for _ in range(n)]; self.arrays=[]; self.threads=[]
  # Allocate/touch before starting the streaming workers, outside measurement.
  for i in range(n):
   src=np.empty(array_bytes,dtype=np.uint8); dst=np.empty_like(src); src.fill(17+i); dst.fill(0); self.arrays.append((src,dst))
  for i in range(n):
   t=threading.Thread(target=self._loop,args=(i,),daemon=True); t.start(); self.threads.append(t)
  self.go.set()
 def _loop(self,i):
  src,dst=self.arrays[i]; self.go.wait()
  while not self.stop.is_set():
   a=time.perf_counter(); np.copyto(dst,src); b=time.perf_counter(); self.events[i].append((a,b))
 def begin_window(self): self.begin=time.perf_counter()
 def end_window(self):
  end=time.perf_counter(); window=end-self.begin; workers=[]
  # np.copyto releases the GIL; appending and reading the small event lists use it.
  # Count only operations fully contained in this timing window. Edge copies are
  # neither prorated nor estimated. Include their count as a boundary uncertainty.
  for i,events in enumerate(self.events):
   complete=[(a,b) for a,b in list(events) if a>=self.begin and b<=end]
   overlapping=[(a,b) for a,b in list(events) if b>self.begin and a<end]
   active=sum(b-a for a,b in complete); count=len(complete); copied=count*self.array_bytes
   workers.append(dict(worker=i,completed_full_copies=count,copy_bytes=copied,read_plus_write_bytes=2*copied,copy_active_s=active,window_s=window,copy_gbps=copied/window/1e9,traffic_gbps=2*copied/window/1e9,active_copy_gbps=copied/active/1e9 if active else 0,active_traffic_gbps=2*copied/active/1e9 if active else 0,completed_edge_overlap_copies=len(overlapping)-count,copy_intervals=[dict(start_s=a-self.begin,end_s=b-self.begin) for a,b in complete]))
  return dict(N=self.n,array_bytes=self.array_bytes,arrays_per_thread=2,window_s=window,workers=workers,cpu_traffic_gbps=sum(w['active_traffic_gbps'] for w in workers),cpu_copy_gbps=sum(w['active_copy_gbps'] for w in workers),cpu_counted_window_traffic_gbps=sum(w['traffic_gbps'] for w in workers),cpu_counted_window_copy_gbps=sum(w['copy_gbps'] for w in workers),traffic_label='Each worker rate is complete measured copy bytes divided by their measured np.copyto elapsed seconds; read+write=2*copy. Only copies wholly inside the row window contribute. Counted-window rates also retained as conservative lower bounds; no partial copy interpolation.',edge_max_omitted_bytes=4*self.n*self.array_bytes)
 def close(self):
  self.stop.set()
  for t in self.threads: t.join()
  self.arrays.clear()
