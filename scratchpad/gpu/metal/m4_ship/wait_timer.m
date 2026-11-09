#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
static IMP original;
static __thread double elapsed;
static void timed_wait(id obj,SEL sel) {
    ((void(*)(id,SEL))original)(obj,sel);
    id<MTLCommandBuffer> cb=obj;
    elapsed=cb.GPUEndTime-cb.GPUStartTime;
}
void install_wait_timer(void) {
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> q=[dev newCommandQueue];
        id<MTLCommandBuffer> cb=[q commandBuffer];Class cls=object_getClass(cb);SEL sel=@selector(waitUntilCompleted);
        original=class_getMethodImplementation(cls,sel);
        class_replaceMethod(cls,sel,(IMP)timed_wait,"v@:");
    }
}
double last_wait_gpu_seconds(void) { return elapsed; }
