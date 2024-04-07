# For now this is a very simple module only made to work with a picoscope 2204A
import serial
import pylsl

import numpy as np

from typing import Callable

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

from enum import IntEnum


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
    b = buffers[3][0:n_values]

    global CHUNK
    CHUNK *= -1
    for va, vb in zip(a, b):
        lsloutlet.push_sample([va, vb * 3, CHUNK * n_values])


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
    stream_name: str = STREAM_NAME, sfreq: int = 10000, n_channels: int = 3
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
    ch_range_b = ps.PS2000_VOLTAGE_RANGE["PS2000_200MV"]

    with ps.open_unit() as device:
        status = setup_osci(
            device, channel_range_a=ch_range_a, channel_range_b=ch_range_b
        )

        # Activate the AWG
        freq_hz = 1.0  # slow frequency for the CorTec device,
        sig_type = 0  # sine wave
        amp_uV = 4_000_000
        status = set_sig_gen(
            status,
            chandle=device.handle,
            freq=freq_hz,
            sig_type=sig_type,
            amp_uv=amp_uV,
        )

        agg_factor = 100
        sample_interval = 100_000  # sample interval in ns
        max_samples = 10_000_000

        """ int16_t ps2000_run_streaming_ns
        (
            int16_t            handle,
            uint32_t           sample_interval,
            PS2000_TIME_UNITS  time_units,
            uint32_t           max_samples,
            int16_t            auto_stop,
            uint32_t           noOfSamplesPerAggregate,
            uint32_t           overview_buffer_size
        );
        """

        res = ps.ps2000_run_streaming_ns(
            device.handle,
            sample_interval,
            TimeUnit.NANOSECOND,
            max_samples,
            False,
            agg_factor,
            50_000,
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
