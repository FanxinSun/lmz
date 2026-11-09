import Foundation
import Metal
let dir=CommandLine.arguments[1], blob=try Data(contentsOf:URL(fileURLWithPath:dir+"/streams.bin")), ref=try Data(contentsOf:URL(fileURLWithPath:dir+"/ref.bin"))
let n=Int(blob.withUnsafeBytes{$0.loadUnaligned(as:UInt32.self)}), plane=Int(blob.withUnsafeBytes{$0.loadUnaligned(fromByteOffset:4,as:UInt32.self)})
let payload=blob.subdata(in:(524+n*16)..<blob.count), offs=blob.subdata(in:524..<(524+n*16)), hdr=blob.subdata(in:8..<524), count=n*plane
let dev=MTLCreateSystemDefaultDevice()!, q=dev.makeCommandQueue()!
let src=try String(contentsOfFile:"lmz/gpu/lmz_rans.metal",encoding:.utf8), lib=try dev.makeLibrary(source:src,options:nil), pso=try dev.makeComputePipelineState(function:lib.makeFunction(name:"lmz_decode_plane_direct")!)
let dst=UnsafeMutableRawPointer.allocate(byteCount:count,alignment:16)
struct Params {var n:UInt32; var plane:UInt32};var params=Params(n:UInt32(n),plane:UInt32(plane))
func now()->Double {ProcessInfo.processInfo.systemUptime}
for run in -2..<7 { autoreleasepool {
 print("{\"event\":\"ready\"}");fflush(stdout);_ = readLine()
 var times:[String:Double]=[:];let all=now();var t=now()
 let padded=NSMutableData(length:payload.count+576)!
 payload.withUnsafeBytes{_ = memcpy(padded.mutableBytes,$0.baseAddress!,payload.count)}
 times["streams_padding_copy_s"]=now()-t;t=now()
 let bs=dev.makeBuffer(bytes:padded.bytes,length:padded.length,options:.storageModeShared)!
 times["streams_mtl_copy_s"]=now()-t;t=now()
 let bo=offs.withUnsafeBytes{dev.makeBuffer(bytes:$0.baseAddress!,length:offs.count,options:.storageModeShared)!}
 let bh=hdr.withUnsafeBytes{dev.makeBuffer(bytes:$0.baseAddress!,length:hdr.count,options:.storageModeShared)!}
 times["metadata_copies_s"]=now()-t;t=now()
 let out=dev.makeBuffer(length:count,options:.storageModeShared)!
 times["output_allocation_s"]=now()-t;t=now()
 let cb=q.makeCommandBuffer()!, enc=cb.makeComputeCommandEncoder()!;enc.setComputePipelineState(pso)
 enc.setBuffer(bs,offset:0,index:0);enc.setBuffer(bo,offset:0,index:1);enc.setBuffer(out,offset:0,index:2);enc.setBuffer(bh,offset:0,index:3);enc.setBytes(&params,length:8,index:4)
 enc.dispatchThreadgroups(MTLSize(width:(n+15)/16,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:128,height:1,depth:1));enc.endEncoding();cb.commit();cb.waitUntilCompleted()
 times["dispatch_wait_s"]=now()-t;times["gpu_s"]=cb.gpuEndTime-cb.gpuStartTime;t=now()
 memcpy(dst,out.contents(),count);times["output_memcpy_s"]=now()-t;times["wall_s"]=now()-all
 print("{\"event\":\"finished\"}");fflush(stdout);_ = readLine()
 let ok=ref.withUnsafeBytes{memcmp(dst,$0.baseAddress!,count)==0}
 let row:[String:Any]=["mode":"old_host_components","run":run,"warmup":run<0,"times":times,"identical":ok]
 print(String(data:try! JSONSerialization.data(withJSONObject:row),encoding:.utf8)!);fflush(stdout);assert(ok)
}}
