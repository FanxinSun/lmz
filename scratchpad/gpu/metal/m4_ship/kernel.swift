// Build with swiftc -O. Drive timing gates through measure_phase4.py.
import Foundation
import Metal
func die(_ s:String)->Never { FileHandle.standardError.write(Data((s+"\n").utf8)); exit(1) }
let args=CommandLine.arguments, dir=args[1], mode=args[2], groups=Int(args[3])!, repeats=Int(args[4])!
let perstream=mode.hasPrefix("perstream"), fused=mode.hasPrefix("fused"), stride=mode.hasSuffix("stride")
let blob=try Data(contentsOf:URL(fileURLWithPath:dir+(perstream ? "/perstream.bin":"/streams.bin")),options:.mappedIfSafe)
let ref=try Data(contentsOf:URL(fileURLWithPath:dir+(fused ? "/bf16.bin":"/ref.bin")),options:.mappedIfSafe)
let n=Int(blob.withUnsafeBytes{$0.loadUnaligned(as:UInt32.self)}), plane=Int(blob.withUnsafeBytes{$0.loadUnaligned(fromByteOffset:4,as:UInt32.self)})
let payloadStart=524+n*16, exponentBytes=n*plane, obytes=exponentBytes*(fused ? 2:1), tpb=perstream ? 32:128
let dev=MTLCreateSystemDefaultDevice()!, queue=dev.makeCommandQueue()!
let source=try String(contentsOfFile:ProcessInfo.processInfo.environment["LMZ_KERNEL_SOURCE"] ?? "lmz/gpu/lmz_rans.metal",encoding:.utf8)
var extra=""
if stride && !(perstream && source.contains("kernel void lmz_decode_perstream_stride")) { extra=try String(contentsOfFile:"scratchpad/gpu/metal/m4_remeasure/"+(perstream || fused ? "wrapper_phase4.metal":"wrapper.metal"),encoding:.utf8) }
let lib=try dev.makeLibrary(source:source+"\n"+extra,options:nil)
let name=perstream ? "lmz_decode_perstream"+(stride ? "_stride":"") : (fused ? "lmz_decode_bf16"+(stride ? "_stride":"") : (stride ? "lmz_decode_plane_stride":"lmz_decode_plane_direct"))
let pso=try dev.makeComputePipelineState(function:lib.makeFunction(name:name)!)
func buffer(_ range:Range<Int>,_ pad:Int=0)->MTLBuffer {
 let b=dev.makeBuffer(length:range.count+pad,options:.storageModeShared)!
 _=blob.withUnsafeBytes{p in memcpy(b.contents(),p.baseAddress!.advanced(by:range.lowerBound),range.count)}
 if pad>0 { memset(b.contents().advanced(by:range.count),0,pad) }; return b
}
let streams=buffer(payloadStart..<blob.count,576), offsets=buffer(524..<payloadStart), header=buffer(8..<524)
let output=dev.makeBuffer(length:obytes,options:.storageModeShared)!
var sm:MTLBuffer?=nil
if fused {
 let data=try Data(contentsOf:URL(fileURLWithPath:dir+"/smplane.bin"),options:.mappedIfSafe)
 sm=data.withUnsafeBytes{dev.makeBuffer(bytes:$0.baseAddress!,length:data.count,options:.storageModeShared)!}
}
struct Params { var nstr:UInt32; var plane:UInt32 }
var params=Params(nstr:UInt32(n),plane:UInt32(plane))
for run in -2..<repeats {
 memset(output.contents(),0xCD,obytes)
 if run==0 { print("{\"event\":\"ready\"}"); fflush(stdout); guard readLine()=="go" else { die("missing timing gate") } }
 let begin=ProcessInfo.processInfo.systemUptime
 let cb=queue.makeCommandBuffer()!, enc=cb.makeComputeCommandEncoder()!
 enc.setComputePipelineState(pso)
 enc.setBuffer(streams,offset:0,index:0); enc.setBuffer(offsets,offset:0,index:1)
 if fused {
  enc.setBuffer(sm,offset:0,index:2); enc.setBuffer(output,offset:0,index:3); enc.setBuffer(header,offset:0,index:4)
  enc.setBytes(&params,length:MemoryLayout<Params>.stride,index:5)
 } else {
  enc.setBuffer(output,offset:0,index:2)
  if perstream { enc.setBytes(&params,length:MemoryLayout<Params>.stride,index:3) }
  else { enc.setBuffer(header,offset:0,index:3); enc.setBytes(&params,length:MemoryLayout<Params>.stride,index:4) }
 }
 enc.dispatchThreadgroups(MTLSize(width:groups,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:tpb,height:1,depth:1))
 enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
 let wall=ProcessInfo.processInfo.systemUptime-begin
 if let e=cb.error { die("GPU error: \(e)") }
 let seconds=cb.gpuEndTime-cb.gpuStartTime
 if seconds<=0 { die("GPU timestamps unavailable") }
 if run==repeats-1 { print("{\"event\":\"finished\"}"); fflush(stdout); guard readLine()=="finish" else { die("missing end timing gate") } }
 let check=run==0 || run==repeats-1
 var ok=true
 if check { ok=ref.withUnsafeBytes{memcmp(output.contents(),$0.baseAddress!,obytes)==0} }
 let row:[String:Any]=["mode":mode,"G":groups,"threads_per_tg":tpb,"run":run,"warmup":run<0,"gpu_s":seconds,"wall_s":wall,"plain_bytes":obytes,"exponent_bytes":exponentBytes,"gbps":Double(obytes)/seconds/1e9,"exponent_gbps":Double(exponentBytes)/seconds/1e9,"verified":check,"identical":ok,"static_threadgroup_bytes":pso.staticThreadgroupMemoryLength,"execution_width":pso.threadExecutionWidth,"device":dev.name,"timestamp":ISO8601DateFormatter().string(from:Date())]
 let json=try JSONSerialization.data(withJSONObject:row,options:.sortedKeys); print(String(data:json,encoding:.utf8)!); fflush(stdout)
 if !ok { die("output mismatch") }
}
