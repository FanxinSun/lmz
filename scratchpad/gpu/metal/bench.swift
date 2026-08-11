// Host harness for lmz's Metal rANS decoder.
//
//     swiftc -O bench.swift -o lmzmetal
//     ./lmzmetal <dir with streams.bin + ref.bin> [smplane.bin bf16.bin]
//
// The shader is compiled from source at runtime, so there is no Xcode project
// and nothing to install beyond the command line tools. Buffers are
// .storageModeShared: on Apple silicon the GPU reads the same physical pages
// as the CPU, so unlike the discrete-GPU case there is no host-to-device copy
// in the load path at all.
import Foundation
import Metal

func die(_ m: String) -> Never {
    FileHandle.standardError.write((m + "\n").data(using: .utf8)!)
    exit(1)
}

func pad(_ s: String, _ n: Int) -> String {
    s.count >= n ? s : s + String(repeating: " ", count: n - s.count)
}
func rpad(_ s: String, _ n: Int) -> String {
    s.count >= n ? s : String(repeating: " ", count: n - s.count) + s
}
func f(_ v: Double, _ d: Int) -> String { String(format: "%.\(d)f", v) }

// Container written by prep_shared.py:
// <u32 nstr><u32 plane><516 B shared table><nstr x (u64 off, u64 len)><payload>
struct Streams {
    let nstr: Int, plane: Int
    let header: Data, offsets: Data, payload: Data
}

func loadStreams(_ path: String) -> Streams {
    guard let d = FileManager.default.contents(atPath: path) else { die("cannot read \(path)") }
    let nstr = Int(d.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self) })
    let plane = Int(d.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self) })
    let iOff = 8 + 516, pOff = 8 + 516 + nstr * 16
    return Streams(nstr: nstr, plane: plane,
                   header: d.subdata(in: 8..<iOff),
                   offsets: d.subdata(in: iOff..<pOff),
                   payload: d.subdata(in: pOff..<d.count))
}

let args = CommandLine.arguments
let dir = args.count > 1 ? args[1] : "."
let st = loadStreams("\(dir)/streams.bin")
guard let refPlane = FileManager.default.contents(atPath: "\(dir)/ref.bin") else {
    die("cannot read \(dir)/ref.bin")
}
let outBytes = st.nstr * st.plane
print("streams  \(st.nstr) x \(st.plane) B = \(f(Double(outBytes)/1e6,1)) MB out, "
    + "\(f(Double(st.payload.count)/1e6,1)) MB compressed")

guard let dev = MTLCreateSystemDefaultDevice() else { die("no Metal device") }
print("device   \(dev.name)  unified=\(dev.hasUnifiedMemory)  "
    + "threadgroupMemory=\(dev.maxThreadgroupMemoryLength) B")

let shaderPath = FileManager.default.fileExists(atPath: "lmz_rans.metal")
    ? "lmz_rans.metal" : "\(dir)/lmz_rans.metal"
guard let src = try? String(contentsOfFile: shaderPath, encoding: .utf8) else {
    die("lmz_rans.metal not found (looked next to the binary and in \(dir))")
}
let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: nil) }
catch { die("shader compile failed:\n\(error)") }
guard let queue = dev.makeCommandQueue() else { die("no command queue") }

func buffer(_ d: Data) -> MTLBuffer {
    d.withUnsafeBytes { p in
        dev.makeBuffer(bytes: p.baseAddress!, length: max(d.count, 1),
                       options: .storageModeShared)!
    }
}

// 64 B of slack: the refill reads a couple of bytes past the final stream,
// exactly as the verified CUDA build does.
var payloadPadded = st.payload
payloadPadded.append(Data(count: 64))

let bStreams = buffer(payloadPadded)
let bOff = buffer(st.offsets)
let bHdr = buffer(st.header)
let bOut = dev.makeBuffer(length: outBytes, options: .storageModeShared)!

struct Params { var nstr: UInt32; var plane: UInt32 }
var P = Params(nstr: UInt32(st.nstr), plane: UInt32(st.plane))

func report(_ name: String, _ secs: Double, _ bytes: Int, _ ok: Bool) {
    print(pad(name, 36) + rpad(f(secs * 1e3, 2) + " ms", 11)
        + rpad(f(Double(bytes) / secs / 1e9, 1) + " GB/s", 13)
        + "  " + (ok ? "byte-identical" : "*** MISMATCH ***"))
}

func runPlane(_ name: String, _ fn: String) {
    guard let fun = lib.makeFunction(name: fn) else { print("\(name): no function \(fn)"); return }
    guard let pso = try? dev.makeComputePipelineState(function: fun) else {
        print("\(name): pipeline creation failed (threadgroup memory?)"); return
    }
    memset(bOut.contents(), 0, outBytes)
    var best = Double.infinity
    for _ in 0..<7 {
        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { die("encoder") }
        enc.setComputePipelineState(pso)
        enc.setBuffer(bStreams, offset: 0, index: 0)
        enc.setBuffer(bOff, offset: 0, index: 1)
        enc.setBuffer(bOut, offset: 0, index: 2)
        enc.setBuffer(bHdr, offset: 0, index: 3)
        enc.setBytes(&P, length: MemoryLayout<Params>.stride, index: 4)
        let threads = st.nstr * 8
        enc.dispatchThreadgroups(
            MTLSize(width: (threads + 127) / 128, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error { print("\(name): \(e)"); return }
        let t = cb.gpuEndTime - cb.gpuStartTime
        if t > 0 && t < best { best = t }
    }
    let got = Data(bytesNoCopy: bOut.contents(), count: refPlane.count, deallocator: .none)
    report(name, best, outBytes, got == refPlane)
}

print("")
print(pad("kernel", 36) + rpad("time", 11) + rpad("throughput", 13) + "  verdict")
runPlane("plane, direct device reads", "lmz_decode_plane_direct")
runPlane("plane, threadgroup prefetch", "lmz_decode_plane_prefetch")

// ---- optional: fused whole-BF16 -----------------------------------------
if args.count > 3 {
    guard let sm = FileManager.default.contents(atPath: args[2]),
          let bf = FileManager.default.contents(atPath: args[3]) else { die("cannot read sm/bf16") }
    let bSm = buffer(sm)
    let bOut2 = dev.makeBuffer(length: outBytes * 2, options: .storageModeShared)!
    guard let fun = lib.makeFunction(name: "lmz_decode_bf16"),
          let pso = try? dev.makeComputePipelineState(function: fun) else { die("bf16 pipeline") }
    var best = Double.infinity
    for _ in 0..<7 {
        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { die("encoder") }
        enc.setComputePipelineState(pso)
        enc.setBuffer(bStreams, offset: 0, index: 0)
        enc.setBuffer(bOff, offset: 0, index: 1)
        enc.setBuffer(bSm, offset: 0, index: 2)
        enc.setBuffer(bOut2, offset: 0, index: 3)
        enc.setBuffer(bHdr, offset: 0, index: 4)
        enc.setBytes(&P, length: MemoryLayout<Params>.stride, index: 5)
        let threads = st.nstr * 8
        enc.dispatchThreadgroups(
            MTLSize(width: (threads + 127) / 128, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error { die("bf16: \(e)") }
        let t = cb.gpuEndTime - cb.gpuStartTime
        if t > 0 && t < best { best = t }
    }
    let got = Data(bytesNoCopy: bOut2.contents(), count: bf.count, deallocator: .none)
    report("fused BF16 (exp + sm raw + merge)", best, bf.count, got == bf)
}
