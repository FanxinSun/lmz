"""D2 aggregation: select the required retry, never select for speed."""
import collections,hashlib,json,statistics,re
from campaign import HERE,ROOT,CACHE

def stats(values):
 values=list(values); return dict(median=statistics.median(values),min=min(values),max=max(values))
def analyze():
 rows=[json.loads(l) for l in (HERE/'runs-phase4.jsonl').open()]; groups=collections.defaultdict(list)
 for r in rows:
  if not r['warmup']: groups[(r['conditions']['campaign'],r['tag'],r['G'],r['attempt'])].append(r)
 attempts=[]
 for (campaign,tag,g,attempt),rs in groups.items():
  r=rs[0]; tpb=r['threads_per_tg']; rate=statistics.median(x['exponent_gbps'] for x in rs)
  cap=30 if r['mode'].startswith('perstream') else 40; resident=min(g,cap); kns=resident*tpb/rate; f=r['gpu_frequency']; attr=r['clock_attribution']
  a=dict(campaign=campaign,tag=tag,mode=r['mode'],G=g,attempt=attempt,n=len(rs),threads_per_tg=tpb,gbps=stats(x['gbps'] for x in rs),exponent_gbps=stats(x['exponent_gbps'] for x in rs),gpu_s=stats(x['gpu_s'] for x in rs),wall_s=stats(x['wall_s'] for x in rs),conditions=r['conditions'],gpu_frequency=f,clock_attribution=attr,foreign_fraction=r['foreign_fraction'],foreign_top=r['foreign_top'],foreign_over_2pct=r['foreign_over_2pct'],static_threadgroup_bytes=r['static_threadgroup_bytes'],all_checked_identical=all(x['identical'] for x in rs),checked_runs=sum(x['verified'] for x in rs),resident_tgs_assumed=resident,resident_cap_assumed=cap,k_ns_apparent=kns,k_cycles_avg=kns*f['f_avg_active_mhz']/1000,k_cycles_attributed=kns*attr['kernel_attributed_mhz']/1000,k_status='rising/first-bend' if g<=cap else 'saturation diagnostic; cannot isolate compute',per_launched_tg_gbps=rate/g,window_s=r['window_s'])
  if 'cpu_contention' in r: a.update(cpu_contention=r['cpu_contention'],traffic_sanity=r['traffic_sanity'])
  attempts.append(a)
 selected={}
 for a in attempts:
  key=(a['campaign'],a['tag'],a['G'])
  if key not in selected or a['attempt']>selected[key]['attempt']: selected[key]=a
 result=dict(machine='Mac mini Apple M4, 10 GPU cores, 24 GiB unified, macOS 15.4.1, AC; VM untouched',run_rows=len(rows),warmup_rows=sum(r['warmup'] for r in rows),timed_rows=sum(not r['warmup'] for r in rows),attempts=attempts,selected=list(selected.values()),shared_prep=json.loads((CACHE/'prep.json').read_text()),perstream_prep=json.loads((CACHE/'perstream-prep.json').read_text()),wrapper_audit=json.loads((HERE/'wrapper-phase4-audit.json').read_text()),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['lmz/gpu/lmz_rans.metal','lmz/gpu/lmzgpu_metal.m']},cleanup=[str(p) for p in CACHE.rglob('*') if p.is_file()],baseline=dict(passed=142,failed=6,total=148))
 for key,path in [('old3b_clock_attribution','clock-attribution-phase4-old3b.json'),('perstream_drift','drift-phase4-perstream.json'),('fused_drift','drift-phase4-fused.json'),('final_tests','tests-phase4.json')]:
  if (HERE/path).exists(): result[key]=json.loads((HERE/path).read_text())
 if (CACHE/'fused-prep.json').exists(): result['fused_prep']=json.loads((CACHE/'fused-prep.json').read_text())
 per=[a for a in result['selected'] if a['tag']=='perstream-sweep' and a['G']<=30]
 if per:
  linear=[a for a in per if a['G']<=10]; slope=sum(a['G']*a['exponent_gbps']['median'] for a in linear)/sum(a['G']**2 for a in linear)
  result['per_chunk_metal_cost_evidence']=dict(unit='lane-cycles per decoded exponent-plane byte',backend='metal',kernel='lmz_decode_perstream',block_threads=32,lanes=8,static_tg_bytes=20992,k_G=[a['G'] for a in per],k_ns_interval=[min(a['k_ns_apparent'] for a in per),max(a['k_ns_apparent'] for a in per)],k_cycles_avg_interval=[min(a['k_cycles_avg'] for a in per),max(a['k_cycles_avg'] for a in per)],k_cycles_attributed_interval=[min(a['k_cycles_attributed'] for a in per),max(a['k_cycles_attributed'] for a in per)],linear_G=[a['G'] for a in linear],slope_gbps_per_tg=slope,max_relative_linear_residual=max(abs(a['exponent_gbps']['median']/(slope*a['G'])-1) for a in linear),resident_tg_cap_assumption=30,occupancy_status='unresolved: prior probe 3 TG/core conflicts with interpreting the high-grid rate using the low-grid slope; 30-TG cap is not established by this sweep',status='measurement evidence only; no shipped cost_model edit')
 memory=[]; seen=set()
 def vm(text): return {m[1]:int(m[2]) for m in re.finditer(r'^([^:\n]+):\s+(\d+)\.',text,re.M)}
 for a in result['selected']:
  if 'cpu_contention' not in a or a['tag'] in seen: continue
  seen.add(a['tag']); before=vm(a['conditions']['memory_stats_before']); after_path=CACHE/('phase4-'+a['tag']+'-memory-after.txt')
  if after_path.exists():
   after=vm(after_path.read_text()); memory.append(dict(tag=a['tag'],global_vm_stat_delta={k:after.get(k,0)-before.get(k,0) for k in ['Pageouts','Swapins','Swapouts','Compressions','Decompressions']}))
 result['cpu_vm_stat_deltas']=memory
 (HERE/'summary-phase4.json').write_text(json.dumps(result,indent=2)); return result
if __name__=='__main__':
 s=analyze(); print(json.dumps({k:s[k] for k in ['run_rows','warmup_rows','timed_rows','per_chunk_metal_cost_evidence'] if k in s},indent=2))
