"""Read-only foreign GPU accounting; per-client identities survive duplicate PIDs."""
import re, subprocess, time

def snapshot(exclude_pid):
 text=subprocess.check_output(['ioreg','-r','-c','AGXDeviceUserClient','-l','-w0'],text=True)
 clients={}
 for block in text.split('+-o ')[1:]:
  ident=re.search(r'id (0x[0-9a-f]+)',block)
  creator=re.search(r'"IOUserClientCreator" = "pid (\d+), ([^"]+)"',block)
  if not ident or not creator or int(creator[1])==exclude_pid: continue
  ns=sum(map(int,re.findall(r'"accumulatedGPUTime"=(\d+)',block)))
  clients[ident[1]]=dict(pid=int(creator[1]),name=creator[2],ns=ns)
 return dict(monotonic_ns=time.monotonic_ns(),clients=clients,total_ns=sum(c['ns'] for c in clients.values()))

def delta(before,after,gpu_s):
 contributions={}; vanished=[]
 for ident,c in after['clients'].items():
  old=before['clients'].get(ident,{}).get('ns',0)
  ns=max(0,c['ns']-old); key=f"{c['pid']} {c['name']}"
  contributions[key]=contributions.get(key,0)+ns
 for ident,c in before['clients'].items():
  if ident not in after['clients']: vanished.append(c)
 total=sum(contributions.values()); top=max(contributions,key=contributions.get) if contributions else None
 return dict(foreign_gpu_ns=total,foreign_fraction=total/(gpu_s*1e9),foreign_top=top,foreign_by_process_ns=contributions,foreign_disappeared_clients=vanished,foreign_snapshot_before=before,foreign_snapshot_after=after,foreign_over_2pct=total>gpu_s*1e9*.02)
