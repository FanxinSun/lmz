"""GPU frequency during a window, from IOReport P-state residency. No sudo, no GPU use.

Written by lmsluice-manager (Claude Code) on 2026-11-09 for the M4 k measurement. It reads
the same counters `macmon` uses: IOReport group "GPU Stats", subgroup "GPU Performance States",
channel "GPUPH". Residency is in timebase ticks (24 MHz on this machine). The P-state ->
frequency table is IORegistry `pmgr` `voltage-states9`, as (Hz u32, mV u32) pairs, entry 0 = OFF.

    from gpufreq import snapshot, delta
    a = snapshot(); ...timed work...; b = snapshot()
    d = delta(a, b)   # {'active_s', 'off_s', 'f_avg_active_mhz', 'f_max_mhz', 'by_state', ...}

`f_avg_active_mhz` is residency-weighted over non-OFF time only. NOTE the table on this M4
is not monotonic in index order (P6 1056, P7 1053, P9 1152, P11 1204, P13 1326); states are
mapped by index exactly as listed, and the table is returned so a reader can check.
"""
import ctypes, re, struct, subprocess, time

_CF = ctypes.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
_IOR = ctypes.CDLL('/usr/lib/libIOReport.dylib')   # lives in the dyld shared cache
_vp = ctypes.c_void_p
_U8 = 0x08000100
for name, res, args in [
    ('CFStringCreateWithCString', _vp, [_vp, ctypes.c_char_p, ctypes.c_uint32]),
    ('CFDictionaryGetValue', _vp, [_vp, _vp]),
    ('CFArrayGetCount', ctypes.c_long, [_vp]),
    ('CFArrayGetValueAtIndex', _vp, [_vp, ctypes.c_long]),
    ('CFStringGetCString', ctypes.c_bool, [_vp, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]),
    ('CFDictionaryCreateMutableCopy', _vp, [_vp, ctypes.c_long, _vp]),
    ('CFRelease', None, [_vp]),
]:
    f = getattr(_CF, name); f.restype = res; f.argtypes = args
for name, res, args in [
    ('IOReportCopyChannelsInGroup', _vp, [_vp, _vp, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]),
    ('IOReportCreateSubscription', _vp, [_vp, _vp, ctypes.POINTER(_vp), ctypes.c_uint64, _vp]),
    ('IOReportCreateSamples', _vp, [_vp, _vp, _vp]),
    ('IOReportCreateSamplesDelta', _vp, [_vp, _vp, _vp]),
    ('IOReportChannelGetChannelName', _vp, [_vp]),
    ('IOReportStateGetCount', ctypes.c_int32, [_vp]),
    ('IOReportStateGetNameForIndex', _vp, [_vp, ctypes.c_int32]),
    ('IOReportStateGetResidency', ctypes.c_int64, [_vp, ctypes.c_int32]),
]:
    f = getattr(_IOR, name); f.restype = res; f.argtypes = args


def _S(s):
    return _CF.CFStringCreateWithCString(None, s.encode(), _U8)


def _str(cfs):
    if not cfs:
        return None
    b = ctypes.create_string_buffer(256)
    _CF.CFStringGetCString(cfs, b, 256, _U8)
    return b.value.decode()


def freq_table_mhz():
    """[(state_index, MHz, mV)] from IORegistry, entry 0 = OFF."""
    out = subprocess.run(['ioreg', '-lw0', '-n', 'pmgr'], capture_output=True, text=True).stdout
    m = re.search(r'"voltage-states9" = <([0-9a-f]+)>', out)
    if not m:
        return []
    b = bytes.fromhex(m.group(1))
    return [(i, struct.unpack_from('<I', b, 8 * i)[0] / 1e6, struct.unpack_from('<I', b, 8 * i + 4)[0])
            for i in range(len(b) // 8)]


_TABLE = freq_table_mhz()
_chans = _IOR.IOReportCopyChannelsInGroup(_S("GPU Stats"), _S("GPU Performance States"), 0, 0, 0)
_mchans = _CF.CFDictionaryCreateMutableCopy(None, 0, _chans)
_subd = _vp()
_sub = _IOR.IOReportCreateSubscription(None, _mchans, ctypes.byref(_subd), 0, None)
if not _sub:
    raise OSError("IOReport subscription to GPU Performance States failed")


def snapshot():
    """An opaque sample plus its wall time."""
    return (_IOR.IOReportCreateSamples(_sub, _subd, None), time.time())


def delta(a, b, timebase_hz=24_000_000):
    d = _IOR.IOReportCreateSamplesDelta(a[0], b[0], None)
    arr = _CF.CFDictionaryGetValue(d, _S("IOReportChannels"))
    by_state = {}
    for i in range(_CF.CFArrayGetCount(arr)):
        ch = _CF.CFArrayGetValueAtIndex(arr, i)
        if _str(_IOR.IOReportChannelGetChannelName(ch)) != "GPUPH":
            continue
        for k in range(_IOR.IOReportStateGetCount(ch)):
            by_state[_str(_IOR.IOReportStateGetNameForIndex(ch, k))] = _IOR.IOReportStateGetResidency(ch, k)
    mhz = {f"P{i}": f for i, f, _mv in _TABLE if i > 0}
    active = {s: t for s, t in by_state.items() if s != "OFF" and t}
    act_ticks = sum(active.values())
    favg = (sum(t * mhz.get(s, 0.0) for s, t in active.items()) / act_ticks) if act_ticks else None
    return {
        "wall_s": b[1] - a[1],
        "active_s": act_ticks / timebase_hz,
        "off_s": by_state.get("OFF", 0) / timebase_hz,
        "f_avg_active_mhz": favg,
        "f_max_mhz": max(mhz.values()) if mhz else None,
        "by_state_s": {s: t / timebase_hz for s, t in active.items()},
        "table_mhz": [round(f) for _i, f, _mv in _TABLE],
        "timebase_hz_assumed": timebase_hz,
    }


if __name__ == "__main__":
    a = snapshot(); time.sleep(1.0); b = snapshot()
    print(delta(a, b))
