"""D2 item 4: re-read old Phase-3b row clocks without altering old records."""
import collections,json,statistics
from campaign import HERE,notice
# This module imports no GPU resources and performs no GPU work.
def attribution(freq,total):
 p15=freq['by_state_s'].get('P15',0); qualifies=p15>=.95*total
 return dict(p15_s=p15,summed_kernel_s=total,p15_to_kernel_ratio=p15/total,p15_ge_95pct=qualifies,kernel_attributed_mhz=freq['table_mhz'][15] if qualifies else freq['f_avg_active_mhz'],kernel_clock_method='P15' if qualifies else 'f_avg_active')
def old_rows():
 groups=collections.defaultdict(list)
 for line in (HERE/'runs.jsonl').open():
  r=json.loads(line)
  if r['conditions']['campaign']=='phase3b-frequency' and not r['warmup']: groups[(r['G'],r['attempt'])].append(r)
 result=[]
 for (g,attempt),rs in sorted(groups.items()):
  rate=statistics.median(r['exponent_gbps'] if 'exponent_gbps' in r else r['gbps'] for r in rs); freq=rs[0]['gpu_frequency']; total=sum(r['gpu_s'] for r in rs); a=attribution(freq,total)
  # 40 resident shared TGs is a curve-derived assumption at/above the bend.
  resident=min(g,40); ns=resident*128/rate
  result.append(dict(G=g,attempt=attempt,n=len(rs),rate_gbps=rate,conditions=rs[0]['conditions'],gpu_frequency=freq,clock_attribution=a,resident_tgs_assumed=resident,resident_lanes_assumed=resident*128,k_ns_apparent=ns,k_cycles_avg=ns*freq['f_avg_active_mhz']/1000,k_cycles_attributed=ns*a['kernel_attributed_mhz']/1000,foreign_fraction=rs[0]['foreign_fraction'],foreign_over_2pct=rs[0]['foreign_over_2pct'],k_status='rising/first-peak cost' if g<=40 else 'saturation diagnostic, not a compute measurement'))
 selected={}
 for r in result:
  if r['G'] not in selected or r['attempt']>selected[r['G']]['attempt']: selected[r['G']]=r
 report=dict(method='P15 if P15_seconds >= 0.95 * summed timed kernel GPU seconds; otherwise f_avg_active',all_attempts=result,selected_rows=list(selected.values()),resident_cap_assumption=40,note='Above 40 TGs apparent k uses the curve-derived cap of 40 resident TGs, and cannot isolate compute cost. No old run or shipped model is edited.')
 (HERE/'clock-attribution-phase4-old3b.json').write_text(json.dumps(report,indent=2))
 return report
if __name__=='__main__':
 r=old_rows(); a=next(x for x in r['selected_rows'] if x['G']==1); b=next(x for x in r['selected_rows'] if x['G']==40)
 notice('D2 item 4',f'Old Phase-3b clock attribution re-read: G=1 {a["k_cycles_avg"]:.2f} average / {a["k_cycles_attributed"]:.2f} attributed; G=40 {b["k_cycles_avg"]:.2f} / {b["k_cycles_attributed"]:.2f} lane-cycles/B. P15 attribution rule and all attempts retained in clock-attribution-phase4-old3b.json. New Phase-4 rows already record the same check.')
