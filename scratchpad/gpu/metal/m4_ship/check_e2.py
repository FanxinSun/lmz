import sys,ctypes,struct,gc,mmap
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'m4_remeasure'));from campaign import locked,ROOT,notice
sys.path.insert(0,str(ROOT));from lmz import gpu,kernels
with locked('ship-e2-smoke'):
 assert gpu.available()[0],gpu.backend()
 exp=bytes([120,121,122,123])*1024;sm=bytes(range(256))*16
 hdr=kernels.rans_shared_header(kernels.rans_freqs(kernels.histogram(exp)));coded=kernels.rans_encode_shared(exp,hdr);off=struct.pack('<QQ',0,len(coded))
 want=kernels.merge_bf16([(exp,0),(sm,0)],len(exp))
 got=gpu.decode_bf16_fused(coded,off,sm,1,len(exp),hdr);assert got==want
 prepared=gpu.prepare_metal_batch(coded,off,1,len(exp),header=hdr,smplane=sm);got=prepared.decode();assert got==want;del prepared;gc.collect();assert got==want
 ordinary=kernels.rans_encode(exp);payload=struct.pack('<II',len(ordinary),len(sm))+ordinary+sm
 from lmz.format import CODEC_BF16
 assert gpu.decode_chunk(payload,CODEC_BF16,2,3,len(exp)*2)==want
 print('fused/retained prepared view/chunk passed',flush=True)
notice('E2 smoke','Fused C/Python API and valid CODEC_BF16 ordinary-header extraction fast path implemented; synthetic fused/chunk identity and prepared output lifetime pass. CODEC_BF16 archives do not carry archive-shared tables; fast path adopts each existing exponent header for its single-stream dispatch, with no format change.')
