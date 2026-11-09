"""Lift shipped bodies verbatim into new stride kernels; never rewrite the decoder."""
import hashlib,json
from campaign import ROOT,HERE
source=(ROOT/'lmz/gpu/lmz_rans.metal').read_text(); records={}; wrappers=[]
for name,groups in [('lmz_decode_perstream',4),('lmz_decode_bf16',16)]:
 start=source.index('kernel void '+name+'('); body_start=source.index('{',start); depth=1; end=body_start+1
 while depth:
  if source[end]=='{': depth+=1
  elif source[end]=='}': depth-=1
  end+=1
 signature=source[start:body_start].rstrip(); body=source[body_start+1:end-1]
 boundary=body.index('    uint gid = tid_g >> 3;'); prefix=body[:boundary]; stream_body=body[boundary+len('    uint gid = tid_g >> 3;'):]
 assert 'threadgroup_barrier' not in stream_body
 signature=signature.replace(name+'(',name+'_stride(')
 closing=signature.rfind(')'); signature=signature[:closing]+',\n    uint grids [[threadgroups_per_grid]])'
 wrapper=signature+'\n{'+prefix+f'    for (uint gid = tid_g >> 3; gid < P.nstr; gid += grids * {groups}) {{'+stream_body+'\n    }\n}\n'
 wrappers.append(wrapper); records[name]=dict(per_stream_sha256=hashlib.sha256(stream_body.encode()).hexdigest(),threadgroup_barriers_in_stream_body=0,changes='gid declaration replaced with grid-stride loop; pre-loop declarations/LUT construction retained verbatim')
(HERE/'wrapper_phase4.metal').write_text('// Generated from runtime shipped source by build_phase4_wrappers.py. Wrapper bodies only.\n\n'+'\n'.join(wrappers))
(HERE/'wrapper-phase4-audit.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),kernels=records),indent=2))
print(json.dumps(records))
