// Build: swiftc -O harness.swift -o ~/Library/Caches/lmz-m4-remeasure/harness
// Run from lmz root: harness DATA_DIR shipped|stride|empty GROUPS RUNS
import Foundation
import Metal
func die(_ s:String)->Never { FileHandle.standardError.write(Data((s+"\n").utf8)); exit(1) }
let args=CommandLine.arguments, dir=args[1], mode=args[2], groups=Int(args[3])!, repeats=Int(args[4])!
let blob=try Data(contentsOf:URL(fileURLWithPath:dir+"/streams.bin"),options:.mappedIfSafe)
let ref=try Data(contentsOf:URL(fileURLWithPath:dir+"/ref.bin"),options:.mappedIfSafe)
let n=Int(blob.withUnsafeBytes{$0.loadUnaligned(as:UInt32.self)}), plane=Int(blob.withUnsafeBytes{$0.loadUnaligned(fromByteOffset:4,as:UInt32.self)})
let payloadStart=524+n*16, obytes=n*plane
let dev=MTLCreateSystemDefaultDevice()!, queue=dev.makeCommandQueue()!
let source=try String(contentsOfFile:"lmz/gpu/lmz_rans.metal",encoding:.utf8)
let extra=try String(contentsOfFile:"scratchpad/gpu/metal/m4_remeasure/wrapper.metal",encoding:.utf8)
// Compile the shipped phase-2 source alone; wrappers are added only in other modes.
let lib=try dev.makeLibrary(source:source+(mode=="shipped" ? "" : "\n"+extra),options:nil)
let name=mode=="shipped" ? "lmz_decode_plane_direct" : (mode=="stride" ? "lmz_decode_plane_stride":"lmz_empty")
let pso=try dev.makeComputePipelineState(function:lib.makeFunction(name:name)!)
func buffer(_ range:Range<Int>,_ pad:Int=0)->MTLBuffer {
 let b=dev.makeBuffer(length:range.count+pad,options:.storageModeShared)!
 blob.withUnsafeBytes{p in memcpy(b.contents(),p.baseAddress!.advanced(by:range.lowerBound),range.count)}
 if pad>0 { memset(b.contents().advanced(by:range.count),0,pad) }; return b
}
let streams=buffer(payloadStart..<blob.count,576), offsets=buffer(524..<payloadStart), header=buffer(8..<524)
let output=dev.makeBuffer(length:obytes,options:.storageModeShared)!
struct Params { var nstr:UInt32; var plane:UInt32 }
var params=Params(nstr:UInt32(n),plane:UInt32(plane))
for run in -2..<repeats {
 if run==0 { print("{\"event\":\"ready\"}"); fflush(stdout); guard readLine()=="go" else { die("missing timing gate") } }
 if mode != "empty" { memset(output.contents(),0xCD,obytes) }
 let begin=ProcessInfo.processInfo.systemUptime
 let cb=queue.makeCommandBuffer()!, enc=cb.makeComputeCommandEncoder()!
 enc.setComputePipelineState(pso)
 enc.setBuffer(streams,offset:0,index:0); enc.setBuffer(offsets,offset:0,index:1)
 enc.setBuffer(output,offset:0,index:2); enc.setBuffer(header,offset:0,index:3)
 enc.setBytes(&params,length:MemoryLayout<Params>.stride,index:4)
 enc.dispatchThreadgroups(MTLSize(width:groups,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:128,height:1,depth:1))
 enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
 let wall=ProcessInfo.processInfo.systemUptime-begin
 if let e=cb.error { die("GPU error: \(e)") }
 let seconds=cb.gpuEndTime-cb.gpuStartTime
 if seconds<=0 { die("GPU timestamps unavailable") }
 let check=mode != "empty" && (run==0 || run==repeats-1)
 var ok=true
 if check { ok=ref.withUnsafeBytes{memcmp(output.contents(),$0.baseAddress!,obytes)==0} }
 let row:[String:Any]=["mode":mode,"G":groups,"run":run,"warmup":run<0,"gpu_s":seconds,"wall_s":wall,"plain_bytes":obytes,"gbps":mode=="empty" ? 0:Double(obytes)/seconds/1e9,"verified":check,"identical":ok,"static_threadgroup_bytes":pso.staticThreadgroupMemoryLength,"execution_width":pso.threadExecutionWidth,"device":dev.name,"timestamp":ISO8601DateFormatter().string(from:Date())]
 let json=try JSONSerialization.data(withJSONObject:row,options:.sortedKeys); print(String(data:json,encoding:.utf8)!); fflush(stdout)
 if !ok { die("output mismatch") }
}
