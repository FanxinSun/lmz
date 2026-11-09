"""Read index bytes directly; no git invocation or index mutation."""
import hashlib,json,os,stat,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]; HERE=Path(__file__).resolve().parent
index=ROOT.parent/'.git/modules/lmz/index'; b=index.read_bytes(); magic,version,count=struct.unpack_from('>4sII',b)
assert magic==b'DIRC' and version in (2,3),(magic,version)
pos=12; mismatches=[]; missing=[]
for i in range(count):
 start=pos; mode=struct.unpack_from('>I',b,pos+24)[0]; expected=b[pos+40:pos+60].hex(); flags=struct.unpack_from('>H',b,pos+60)[0]; pos+=62
 if version>=3 and flags&0x4000: pos+=2
 end=b.index(b'\0',pos); name=b[pos:end].decode(); pos=start+((end-start+1+7)//8)*8
 if mode&0o170000==0o160000: continue
 p=ROOT/name
 if not p.exists() and not p.is_symlink(): missing.append(name); continue
 raw=os.readlink(p).encode() if p.is_symlink() else p.read_bytes()
 actual=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
 if actual!=expected: mismatches.append(name)
result=dict(index_version=version,index_entries=count,content_mismatches=mismatches,missing=missing,method='read-only Git index parsing and blob hashing; no git command')
phase=sys.argv[1] if len(sys.argv)>1 else 'initial'
report_path=ROOT/'docs/evaluations/m4-metal-decoder-remeasure.md'
raw=report_path.read_bytes()
if phase=='initial':
 result['report_prefix_length']=len(raw); result['report_prefix_sha256']=hashlib.sha256(raw).hexdigest()
 assert not mismatches and not missing,result
else:
 before=json.loads((HERE/'tracked-phase4-initial.json').read_text())
 result['report_prefix_preserved']=hashlib.sha256(raw[:before['report_prefix_length']]).hexdigest()==before['report_prefix_sha256']
 assert set(mismatches)<=set(['docs/evaluations/m4-metal-decoder-remeasure.md']) and not missing and result['report_prefix_preserved'],result
(HERE/('tracked-phase4-'+phase+'.json')).write_text(json.dumps(result,indent=2)); print(json.dumps(result))
