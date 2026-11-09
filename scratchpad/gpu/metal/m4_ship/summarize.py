import ast,collections,hashlib,json,statistics,struct
from pathlib import Path
R=Path(__file__).resolve().parents[4];H=Path(__file__).resolve().parent;C=Path.home()/'Library/Caches/lmz-m4-remeasure'
def read(name):return [json.loads(l) for l in (H/name).read_text().splitlines()] if (H/name).exists() else []
def stats(rs,key):
 v=[r[key] for r in rs];return dict(median=statistics.median(v),min=min(v),max=max(v))
def kernel(name):
 rows=read(name);groups=collections.defaultdict(list)
 for r in rows:
  if not r['warmup']:groups[(r['tag'],r['G'],r['attempt'])].append(r)
 chosen={}
 for (tag,g,attempt),rs in groups.items():
  x=dict(tag=tag,G=g,attempt=attempt,n=len(rs),gbps=stats(rs,'gbps'),gpu_s=stats(rs,'gpu_s'),clock=rs[0]['clock_attribution'],foreign_fraction=rs[0]['foreign_fraction'],foreign_top=rs[0]['foreign_top'],identical=all(r['identical'] for r in rs if r.get('verified',True)))
  if (tag,g) not in chosen or attempt>chosen[tag,g]['attempt']:chosen[tag,g]=x
 return list(chosen.values())
def api(name):
 groups=collections.defaultdict(list)
 for r in read(name):
  if not r['warmup']:groups[r['mode'],r['attempt']].append(r)
 selected={}
 for (mode,attempt),rs in groups.items():
  x=dict(mode=mode,attempt=attempt,n=len(rs),api_gbps=stats(rs,'gbps'),kernel_gbps=stats(rs,'kernel_gbps'),wall_s=stats(rs,'wall_s'),verified_timed_runs=sum(r.get('verified',True) for r in rs),check_all=rs[0].get('check_all',True),tuned_tgs=rs[0].get('tuned_tgs'),prepare_s=rs[0].get('prepare_s'),foreign_fraction=[r['foreign']['foreign_fraction'] for r in rs],identical=all(r['identical'] for r in rs if r.get('verified',True)))
  if mode not in selected or attempt>selected[mode]['attempt']: selected[mode]=x
 return list(selected.values())
result=dict(final_strategy='B1: exact HEAD refill bodies, host mapped-tail slack gate or zero-padded copy; E1 stride padded to complete SIMD groups; ARC enabled',kernel_b1=kernel('kernel-b1-runs.jsonl'),kernel_b2_rejected=kernel('kernel-b2-runs.jsonl'),api_b1=api('api-b1-runs.jsonl'),api_b1_full_traffic_retuned=api('api-b1-retuned-runs.jsonl'),api_b1_prepared_resident=api('api-b1-resident-runs.jsonl'),api_b1_extra=api('api-b1-extra-runs.jsonl'))
chunks=collections.defaultdict(list)
for r in read('chunk-runs.jsonl'):
 if not r['warmup']:chunks[r['bf16_bytes'],r['mode']].append(r)
result['chunk_api']=[dict(bf16_bytes=size,mode=mode,gbps=stats(rs,'gbps'),wall_s=stats(rs,'wall_s'),identical=all(r['identical'] for r in rs if r.get('verified',True))) for (size,mode),rs in chunks.items()]
result['kernel_b1_comparison']=[]
for a in result['kernel_b1']:
 if a['tag'].startswith('b1-'):
  b=next(x for x in result['kernel_b1'] if x['tag']=='head-'+a['tag'][3:] and x['G']==a['G']);result['kernel_b1_comparison'].append(dict(mode=a['tag'][3:],G=a['G'],HEAD_gbps=b['gbps']['median'],B1_gbps=a['gbps']['median'],ratio=a['gbps']['median']/b['gbps']['median']))
profile=read('profile-runs.jsonl');profile=[r for r in profile if not r['warmup'] and r['attempt']==max(x['attempt'] for x in profile)]
result['old_host_profile_stage_s']={k:statistics.median(r['times'][k] for r in profile) for k in profile[0]['times']} if profile else {}
for key,name in [('tests','test-results-repeat.json'),('protected_partial_tests','audit-tests.txt'),('real_settings','real-settings-check.json'),('existing_exports','existing-exports.json')]:
 if (H/name).exists():result[key]=json.loads((H/name).read_text()) if name.endswith('.json') else (H/name).read_text()
head=(C/'head/lmz_rans.metal').read_bytes();source=(R/'lmz/gpu/lmz_rans.metal').read_bytes();result['HEAD_kernel_prefix_exact']=source.startswith(head)
# Read-only index inventory: no git invocation/state change.
b=(R.parent/'.git/modules/lmz/index').read_bytes();magic,version,count=struct.unpack_from('>4sII',b);assert version==2 and magic==b'DIRC';pos=12;changed=[]
for _ in range(count):
 start=pos;expected=b[pos+40:pos+60].hex();mode=struct.unpack_from('>I',b,pos+24)[0];pos+=62;end=b.index(b'\0',pos);name=b[pos:end].decode();pos=start+((end-start+1+7)//8)*8
 if mode&0o170000==0o160000:continue
 p=R/name
 if not p.exists():changed.append(name);continue
 data=p.read_bytes();actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
 if actual!=expected:changed.append(name)
result['tracked_files_changed']=changed
allowed={'lmz/gpu/lmz_rans.metal','lmz/gpu/lmzgpu_metal.m','lmz/gpu/__init__.py','lmz/gpu/build.py','tests/test_lmz.py'}
result['files_written_by_this_task']=sorted(allowed)
result['other_observed_tracked_changes']=sorted(set(changed)-allowed)
assert 'lmz/gpu/lmzgpu.cu' not in changed
result['CUDA_source_unchanged']='lmz/gpu/lmzgpu.cu' not in changed
result['docs_and_cost_notes']='Docs/README/limitations intentionally untouched per final manager instruction. Published k values unchanged because B1 restores original hot loops; manager owns provenance/geometry notes.'
result['new_files']=[str(p.relative_to(R)) for p in H.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
result['cache_cleanup']=[str(p) for p in C.rglob('*') if p.is_file()]
result['limitations']=['Tuning runs once per process on the first substantial actual ordinary batch; five device-core-scaled/full-grid candidates, four launches each, every full output verified in bounded chunks. Preparation/cold tuning cost is separate from warm rates.','Most rate rows remain flagged for desktop foreign GPU activity; both required attempts retained, second attempt used regardless of speed.','Prepared results share output; next decode overwrites earlier views. Inputs and mapped page tails must remain valid/unchanged until bindings/views are released.','B1 trusts valid streams and mapped padding like CUDA; corrupt streams can fail a command buffer and return None, and caller-owned output can be partially modified on failure.','HEAD device ABI real-32KiB workload not safely benchmarkable because its declared coded extent is 128*nstr+pad; only a correctly fitting small-plane workload is tested.','Single-chunk fused dispatch remains opt-in; measured much slower than CPU native path at all three real chunk sizes.','Fused API wait/host overhead remains substantially larger than kernel time on the 1.87GB output; timings are reported without attributing undocumented driver behavior.']
(H/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['tracked_files_changed','HEAD_kernel_prefix_exact','CUDA_source_unchanged','kernel_b1_comparison']},indent=2))
