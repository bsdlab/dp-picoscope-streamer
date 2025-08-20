# For now this is a very simple module only made to work with a picoscope 2204A
import threading
from ctypes import POINTER, c_int16, c_uint32
from enum import IntEnum
from functools import partial

import pylsl
from fire import Fire
from picosdk.ctypes_wrapper import C_CALLBACK_FUNCTION_FACTORY
from picosdk.device import Device
from picosdk.functions import assert_pico2000_ok
from picosdk.PicoDeviceEnums import picoEnum
from picosdk.ps2000 import (
    ps2000 as ps,
)  # although my osci is a 2204a, it requires the older ps2000 SDK - see here: https://www.picotech.com/support/viewtopic.php?t=13171


class TimeUnit(IntEnum):
    FEMTOSECOND = 0
    PICOSECOND = 1
    NANOSECOND = 2
    MICROSECOND = 3
    MILLISECOND = 4
    SECOND = 5


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
LASTMAX = 0
CHUNK = 1  # 10_000


def get_data(
    buffers,
    _overflow,
    _triggered_at,
    _triggered,
    _auto_stop,
    n_values,
    lsloutlet: pylsl.StreamOutlet,
):
    # Threaded execution is not a good idead at all -> very large gaps
    # THREADPOOL.submit(partial(buffer_to_lsl, buffers, n_values, lsloutlet))
    buffer_to_lsl(buffers, n_values, lsloutlet)


def buffer_to_lsl(buffers, n_values, lsloutlet: pylsl.StreamOutlet):
    a = buffers[0][0:n_values]
    global CHUNK
    CHUNK *= -1

    try:
        lsloutlet.push_chunk(a)
    except ValueError:
        pass
    except SystemError:
        pass


def setup_osci(device: Device, channel_range_a: int = 3, channel_range_b: int = 3):
    status = {}
    status["setChA"] = ps.ps2000_set_channel(
        device.handle,
        0,
        picoEnum.PICO_CHANNEL["PICO_CHANNEL_A"],
        1,
        channel_range_a,
    )

    assert_pico2000_ok(status["setChA"])
    return status


def update(device: Device, callback: None = None):
    ps.ps2000_get_streaming_last_values(device.handle, callback)


def get_stream_outlet(
    stream_name: str = STREAM_NAME,
    sfreq: int = 10000,
    n_channels: int = 1,
) -> pylsl.StreamOutlet:
    info = pylsl.StreamInfo(
        stream_name,
        "EEG",
        n_channels,
        sfreq,
        "float32",
        f"{stream_name}",
    )

    max_buffered_s = 5

    return pylsl.StreamOutlet(info, max_buffered=max_buffered_s)


def main(stop_event: threading.Event = threading.Event()):
    ch_range_a = ps.PS2000_VOLTAGE_RANGE["PS2000_200MV"]

    with ps.open_unit() as device:
        status = setup_osci(device, channel_range_a=ch_range_a)

        agg_factor = 1
        sample_interval = 300  # sample interval in ns - try to be as fast as possible, processing is usually sample_interval * 500
        max_samples = 10_000

        res = ps.ps2000_run_streaming_ns(
            device.handle,
            sample_interval,
            TimeUnit.NANOSECOND,
            max_samples,  # seems not to really have an impact -> the sdk seems to always work in chunks of 500 if possible, no matter what time etc is specified
            False,
            agg_factor,
            25_000,
        )

        lsloutlet = get_stream_outlet(
            stream_name=STREAM_NAME,
            sfreq=1 / (agg_factor * sample_interval * 1e-9),
        )

        pget_data = partial(get_data, lsloutlet=lsloutlet)
        cback = CALLBACK(pget_data)
        pupdate = partial(update, device=device, callback=cback)

        while True and device.handle is not None and not stop_event.is_set():
            # fetch data from the osci
            pupdate()


def get_main_thread() -> tuple[threading.Thread, threading.Event]:
    stop_event = threading.Event()
    stop_event.clear()

    thread = threading.Thread(target=main, kwargs={"stop_event": stop_event})
    thread.start()

    return thread, stop_event


if __name__ == "__main__":
    Fire(main)
