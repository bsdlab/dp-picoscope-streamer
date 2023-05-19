# For now this is a very simple module only made to work with a picoscope 2204A
import mmap
import pylsl

from pathlib import Path
from fire import Fire
import time

import threading

from functools import partial

from picosdk.ps2000 import (
    ps2000 as ps,
)  # although my osci is a 2204a, it requires the older ps2000 SDK - see here: https://www.picotech.com/support/viewtopic.php?t=13171

from picosdk.functions import assert_pico2000_ok
from picosdk.ctypes_wrapper import C_CALLBACK_FUNCTION_FACTORY
from picosdk.device import Device
from picosdk.PicoDeviceEnums import picoEnum

from picoscope_streamer.awg import set_sig_gen

from ctypes import POINTER, c_int16, c_uint32

CALLBACK = C_CALLBACK_FUNCTION_FACTORY(
    None,
    POINTER(POINTER(c_int16)),
    c_int16,
    c_uint32,
    c_int16,
    c_int16,
    c_uint32,
)

STREAM_NAME = "PICOSTREAM"

MMAP_FILE = "../offset_utf8.txt"


# In [37]: %timeit read_mmap_from_start(mmapo)
# 136 ns ± 0.904 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops e
# ach)
# about 50ns is for the int() conversion
def read_mmap_from_start(mmapo: mmap.mmap):
    mmapo.seek(0)
    return int(mmapo.read())


def get_data(
    buffers,
    _overflow,
    _triggered_at,
    _triggered,
    _auto_stop,
    n_values,
    lsloutlet: pylsl.StreamOutlet | None = None,
    mmapo: mmap.mmap | None = None,
):
    # From Docs: https://www.picotech.com/download/manuals/ps2000pg.en-10.pdf
    # overviewBuffer [0] - ch_a_max
    # overviewBuffer [1] - ch_a_min
    # overviewBuffer [2] - ch_b_max
    # overviewBuffer [3] - ch_b_min

    a = buffers[0][0:n_values]
    b = buffers[3][0:n_values]

    # b = read_mmap_from_start(mmapo) if mmapo else 0
    # b = 0

    for va, vb in zip(a, b):
        # print(f"{va=}, {vb=}")
        lsloutlet.push_sample([va, vb * 3])


def setup_osci(
    device: Device, channel_range_a: int = 3, channel_range_b: int = 3
):
    status = {}
    status["setChA"] = ps.ps2000_set_channel(
        device.handle,
        0,
        picoEnum.PICO_CHANNEL["PICO_CHANNEL_A"],
        1,
        channel_range_a,
    )
    status["setChB"] = ps.ps2000_set_channel(
        device.handle,
        0,
        picoEnum.PICO_CHANNEL["PICO_CHANNEL_B"],
        1,
        channel_range_b,
    )
    assert_pico2000_ok(status["setChA"])
    assert_pico2000_ok(status["setChB"])
    return status


def update(device: Device, callback: None = None):
    ps.ps2000_get_streaming_last_values(device.handle, callback)


def get_stream_outlet(
    stream_name: str = STREAM_NAME, sfreq: int = 10000
) -> pylsl.StreamOutlet:
    n_channels = 2
    info = pylsl.StreamInfo(
        stream_name,
        "EEG",
        n_channels,
        sfreq,
        "float32",
        f"{stream_name}",
    )

    max_buffered_s = 10

    return pylsl.StreamOutlet(info, max_buffered=max_buffered_s)


def main(stop_event: threading.Event = threading.Event()):
    global adc_samples, nnew
    adc_samples = []
    ch_range_a = ps.PS2000_VOLTAGE_RANGE["PS2000_1V"]
    ch_range_b = ps.PS2000_VOLTAGE_RANGE["PS2000_1V"]

    with ps.open_unit() as device:
        status = setup_osci(
            device, channel_range_a=ch_range_a, channel_range_b=ch_range_b
        )

        # Activate the AWG
        freq_hz = 1.0  # we expect reaction to be faster than 10ms
        sig_type = 2  # Triangular is better for buffer stability (the edge in the saber tooth leads to problems)
        amp_uV = 2_000_000
        status = set_sig_gen(
            status,
            chandle=device.handle,
            freq=freq_hz,
            sig_type=sig_type,
            amp_uv=amp_uV,
        )

        agg_factor = 100
        sample_interval = 1000  # sample interval in ns
        max_samples = 1_000_000

        res = ps.ps2000_run_streaming_ns(
            device.handle,
            sample_interval,
            2,
            max_samples,
            False,
            agg_factor,
            1_000_000,
        )

        lsloutlet = get_stream_outlet(
            stream_name=STREAM_NAME,
            sfreq=1 / (agg_factor * sample_interval * 1e-9),
        )

        # if Path(MMAP_FILE).exists():
        #     with open(MMAP_FILE, "r", encoding="utf-8") as fo:
        #         mmapo = mmap.mmap(
        #             fo.fileno(), length=0, access=mmap.ACCESS_READ
        #         )
        # else:
        #     mmapo = None
        mmapo = None

        pget_data = partial(get_data, lsloutlet=lsloutlet, mmapo=mmapo)
        cback = CALLBACK(pget_data)
        pupdate = partial(update, device=device, callback=cback)

        while True and device.handle is not None and not stop_event.is_set():
            # fetch data from the osci
            pupdate()

        # mmapo.close()


def get_main_thread() -> tuple[threading.Thread, threading.Event]:
    stop_event = threading.Event()
    stop_event.clear()

    thread = threading.Thread(target=main, kwargs={"stop_event": stop_event})
    thread.start()

    return thread, stop_event


if __name__ == "__main__":
    Fire(main)
