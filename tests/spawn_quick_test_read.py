import time
from ctypes import POINTER, c_int16, c_uint32
from functools import partial

import pylsl
from picosdk.ctypes_wrapper import C_CALLBACK_FUNCTION_FACTORY
from picosdk.device import Device
from picosdk.functions import assert_pico2000_ok
from picosdk.PicoDeviceEnums import picoEnum
from picosdk.ps2000 import ps2000

from picoscope_streamer.awg import set_sig_gen

CALLBACK = C_CALLBACK_FUNCTION_FACTORY(
    None,
    POINTER(POINTER(c_int16)),
    c_int16,
    c_uint32,
    c_int16,
    c_int16,
    c_uint32,
)


TICK = 0
lastp = 0


def get_stream_outlet(
    stream_name: str = "test", sfreq: int = 100_000, n_channels: int = 1
) -> pylsl.StreamOutlet:
    info = pylsl.StreamInfo(
        stream_name,
        "EEG",
        n_channels,
        sfreq,
        "float32",
        f"{stream_name}",
    )

    max_buffered_s = 1

    return pylsl.StreamOutlet(info, max_buffered=max_buffered_s)


def get_overview_buffers(
    buffers, _overflow, _triggered_at, _triggered, _auto_stop, n_values, outlets
):
    tstamp = pylsl.local_clock()

    global TICK, lastp
    valsa = buffers[0][0:n_values]
    valsb = buffers[2][0:n_values]
    acquire_offset_s = 500 * 100 * len(valsa) / 10**9
    #
    # if max(valsb) > 1000 and TICK - lastp > 100:
    #     print(f"Tick: {TICK} - {time.time_ns()} - {len(valsb)}")
    #     lastp = TICK

    outlets[0].push_chunk(valsb, tstamp)
    outlets[1].push_chunk(valsb, tstamp - acquire_offset_s)
    TICK += 1


def setup_osci(device: Device, channel_range_a: int = 3, channel_range_b: int = 3):
    status = {}
    status["setChA"] = ps2000.ps2000_set_channel(
        device.handle,
        0,
        picoEnum.PICO_CHANNEL["PICO_CHANNEL_A"],
        1,
        channel_range_a,
    )
    status["setChB"] = ps2000.ps2000_set_channel(
        device.handle,
        0,
        picoEnum.PICO_CHANNEL["PICO_CHANNEL_B"],
        1,
        channel_range_b,
    )
    assert_pico2000_ok(status["setChA"])
    assert_pico2000_ok(status["setChB"])
    return status


ch_range_a = ps2000.PS2000_VOLTAGE_RANGE["PS2000_1V"]
ch_range_b = ps2000.PS2000_VOLTAGE_RANGE["PS2000_1V"]
with ps2000.open_unit() as device:
    print("Device info: {}".format(device.info))
    status = setup_osci(device, channel_range_a=ch_range_a, channel_range_b=ch_range_b)
    # Activate the AWG
    freq_hz = 1000.0  # we expect reaction to be faster than 10ms
    sig_type = 0  # Triangular is better for buffer stability (the edge in the saber tooth leads to problems)
    amp_uV = 2_000_000
    status = set_sig_gen(
        status,
        chandle=device.handle,
        freq=freq_hz,
        sig_type=sig_type,
        amp_uv=amp_uV,
    )

    # res = ps2000.ps2000_set_channel(
    #     device.handle,
    #     picoEnum.PICO_CHANNEL["PICO_CHANNEL_A"],
    #     True,
    #     picoEnum.PICO_COUPLING["PICO_DC"],
    #     ps2000.PS2000_VOLTAGE_RANGE["PS2000_5V"],
    # )
    # assert_pico2000_ok(res)

    res = ps2000.ps2000_run_streaming_ns(
        device.handle, 500, 2, 10_000, False, 100, 25_000
    )
    assert_pico2000_ok(res)

    start_time = time.time_ns()

    outlet_1 = get_stream_outlet(stream_name="test_a")
    outlet_2 = get_stream_outlet(stream_name="test_b")
    cb = partial(get_overview_buffers, outlets=[outlet_1, outlet_2])

    callback = CALLBACK(cb)
    while 1 == 1:
        ps2000.ps2000_get_streaming_last_values(device.handle, callback)
