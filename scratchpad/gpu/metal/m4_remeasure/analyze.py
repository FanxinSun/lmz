"""Summarize retained attempts; choose the required retry, never the fastest attempt."""
import collections, hashlib, json, statistics
from pathlib import Path
from campaign import HERE,ROOT,CACHE

def stats(rs,field):
 vals=[r[field] for r in rs if field in r]; return dict(median=statistics.median(vals),min=min(vals),max=max(vals)) if vals else None
rows=[json.loads(s) for s in (HERE/'runs.jsonl').read_text().splitlines()]
groups=collections.defaultdict(list)
for r in rows:
 if r.get('warmup') is False:
  key=(r['conditions']['campaign'],r['mode'],r.get('G'),r.get('attempt',0),r.get('workload','real_qwen3'))
  groups[key].append(r)
attempts=[]
for (campaign,mode,g,attempt,workload),rs in groups.items():
 attempts.append(dict(campaign=campaign,mode=mode,G=g,attempt=attempt,workload=workload,n=len(rs),gbps=stats(rs,'gbps'),gpu_s=stats(rs,'gpu_s'),wall_s=stats(rs,'wall_s'),foreign_fraction=rs[0].get('foreign_fraction'),foreign_top=rs[0].get('foreign_top'),foreign_over_2pct=rs[0].get('foreign_over_2pct'),gpu_frequency=rs[0].get('gpu_frequency'),k_ns=stats(rs,'k_ns'),k_cycles=stats(rs,'k_cycles'),all_checked_identical=all(r['identical'] for r in rs),checked_runs=sum(r.get('verified',False) for r in rs)))
selected={}
for a in attempts:
 key=(a['campaign'],a['mode'],a['G'],a['workload'])
 if key not in selected or a['attempt']>selected[key]['attempt']: selected[key]=a
sweep=sorted((a for a in selected.values() if a['mode']=='stride' and not a['campaign'].startswith('phase3b')),key=lambda a:a['G'])
for a in sweep:
 a['per_tg_gbps']=a['gbps']['median']/a['G']; a['k_ns_if_G_resident']=128/a['per_tg_gbps'] if a['G']<=30 else None
prep=json.loads((CACHE/'prep.json').read_text())
summary=dict(machine='Apple M4 Mac mini, 10 GPU cores, 24 GiB unified',git_sha='f39a8d47714480a080f49ca9dcd3870eb7026be9',workload=prep,attempts=attempts,selected_attempts=list(selected.values()),sweep=sweep,verify=json.loads((HERE/'verify.json').read_text()),drift=json.loads((HERE/'drift.json').read_text()),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['lmz/gpu/lmz_rans.metal','lmz/gpu/lmzgpu_metal.m','lmz/gpu/lmzgpu.cu']},baseline=dict(passed=141,failed=6,total=147),traffic_factor=1+1/prep['expansion'],clock_probe_available=any(p.is_file() for p in (ROOT.parent/'lmsluice/results/probe_m4').glob('*')),cleanup=[str(p) for p in CACHE.rglob('*') if p.is_file()])
linear=[a for a in sweep if a['G']<=12]
slope=sum(a['G']*a['gbps']['median'] for a in linear)/sum(a['G']**2 for a in linear)
strict=[a['k_ns_if_G_resident'] for a in linear]
broad=[a['k_ns_if_G_resident'] for a in sweep if a['G']<=30]
full=next(a for a in sweep if a['G']==1787)
summary['cost_derivation']=dict(linear_G=[a['G'] for a in linear],origin_fit_slope_gbps_per_tg=slope,max_relative_fit_residual=max(abs(a['gbps']['median']/(a['G']*slope)-1) for a in linear),k_ns_interval=[min(strict),max(strict)],k_ns_broader_pre_peak_sensitivity=[min(broad),max(broad)],first_peak_G=40,inferred_resident_tgs_per_core=4,residency_status='tentative inference from first rate peak; not a hardware residency counter',first_peak_rate_gbps=next(a['gbps']['median'] for a in sweep if a['G']==40),full_grid_rate_gbps=full['gbps']['median'],achieved_fraction=full['gbps']['median']*summary['traffic_factor']/120,k_cycles=None,clock_status='no clock probe at requested path at report generation',hygiene_status='every amended sweep row and its retry exceeds 2% foreign accumulated GPU time; conditional estimates only')
summary['conditions']={p.stem:json.loads(p.read_text()) for p in CACHE.glob('*-conditions.json')}
summary['run_conditions']={r['conditions']['timestamp']:r['conditions'] for r in rows}
summary['frequency_sweep']=sorted((a for a in selected.values() if a['campaign'].startswith('phase3b')),key=lambda a:a['G'])
if summary['frequency_sweep']:
 fr=summary['frequency_sweep']; pre=[a for a in fr if a['G']<=30]; clean=[a for a in pre if not a['foreign_over_2pct']]
 summary['frequency_derivation']=dict(method='measured residency-weighted GPU frequency over active states from IOReport GPUPH',pstate_table_source='IORegistry pmgr voltage-states9, indexed in non-monotonic order',timebase_hz_assumed=24000000,pre_saturation_G=[a['G'] for a in pre],k_cycles_interval=[min(a['k_cycles']['median'] for a in pre),max(a['k_cycles']['median'] for a in pre)],k_ns_interval=[min(a['k_ns']['median'] for a in pre),max(a['k_ns']['median'] for a in pre)],clean_G=[a['G'] for a in clean],clean_k_cycles_row_median_interval=[min(a['k_cycles']['median'] for a in clean),max(a['k_cycles']['median'] for a in clean)],strict_linear_k_cycles_interval=[min(a['k_cycles']['median'] for a in pre if a['G']<=10),max(a['k_cycles']['median'] for a in pre if a['G']<=10)],label='conditional broader interval; only G=1,2 meet foreign-time criterion')
 summary['cost_derivation']['k_cycles']=summary['frequency_derivation']['k_cycles_interval']
 summary['cost_derivation']['clock_status']='Phase 3b per-row IOReport residency-weighted active-state MHz, not FMA proxy'
 summary['cost_derivation']['hygiene_status']='Original sweep all flagged; Phase 3b G=1,2 below 2%, higher G remain flagged after retry'
 summary['row_frequency_available']=True
summary['tracked_content_check']=json.loads((HERE/'tracked-content-check.json').read_text())
summary['run_row_count']=len(rows)
if (HERE/'final-tests.json').exists(): summary['final_tests']=json.loads((HERE/'final-tests.json').read_text())
(HERE/'summary.json').write_text(json.dumps(summary,indent=2))
for a in sweep: print(a['G'],f"{a['gbps']['median']:.4f}",f"{a['per_tg_gbps']:.5f}",a['k_ns_if_G_resident'],a['foreign_fraction'])
