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

from dareplane_utils.general.time import sleep_s


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

TARGET_SRATE_HZ = 5_000

LSL_BUFFER = np.zeros((5_000_000, 3))
LSL_LAST_PUSH = time.perf_counter_ns()
CURRIDX = 0
LAST_CB_CALL = time.perf_counter_ns()


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

    # global LAST_CB_CALL

    # print(f"Last CB: {(time.perf_counter_ns() - LAST_CB_CALL)*1e-6:.3f}ms")
    # LAST_CB_CALL = time.perf_counter_ns()
    buffer_to_lsl(buffers, n_values, lsloutlet)


def buffer_to_lsl(buffers, n_values, lsloutlet: pylsl.StreamOutlet):
    a = buffers[0][0:n_values]
    b = buffers[3][0:n_values]

    global CHUNK, FROM_LAST_BUFFER, TARGET_SRATE_HZ, LSL_LAST_PUSH, CURRIDX
    # CHUNK *= -1

    # put to b
    LSL_BUFFER[CURRIDX : CURRIDX + 1, 0] = np.mean(a)
    LSL_BUFFER[CURRIDX : CURRIDX + 1, 1] = np.mean(b * 3)
    CURRIDX += 1
    # LSL_BUFFER[CURRIDX : CURRIDX + n_values, 2] = CHUNK * n_values

    dt = (time.perf_counter_ns() - LSL_LAST_PUSH) * 1e-9
    req_samples = int(TARGET_SRATE_HZ * dt)

    LSL_BUFFER[CURRIDX : CURRIDX + 1, 2] = req_samples

    # just start over again, because here a long period was missing
    if req_samples >= 2 * TARGET_SRATE_HZ:
        # print(f"Starting over - {req_samples=}, {LSL_LAST_PUSH=}")
        CURRIDX = 0
        LSL_LAST_PUSH = time.perf_counter_ns()
    else:
        # print(f"{CURRIDX=}, \n{LSL_BUFFER[:CURRIDX].mean(axis=0)=}")

        if req_samples >= 1:

            LSL_LAST_PUSH = time.perf_counter_ns()

            # print(f"Pushing: {req_samples=}, {dt=}")

            # just push rectified values by mean
            pre_push = time.perf_counter_ns()
            for _ in range(req_samples):
                lsloutlet.push_chunk(list(LSL_BUFFER[:CURRIDX].mean(axis=0)))
            CURRIDX = 0

            # print(
            #     f"Push took: {(time.perf_counter_ns() - pre_push) * 1e-6:.3f}ms"
            # )
        # else:
        #     print(f"Not pushing")
    # for va, vb in zip(a, b):
    #     lsloutlet.push_sample([va, vb * 3, CHUNK * n_values])


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
    stop_event = threading.Event() if stop_event is None else stop_event
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

        # tested with 'tests/plot_test_with_awg.py' this seems to be a good config for plainly running to a local buffer
        # maybe increase agg_factor to about 5 for approx 1kHz dumps of data
        agg_factor = (
            5  # agg_factor = 3 seems to be too much load for the lenovo
        )

        sample_interval = (
            300  # sample interval in ns - 300 was too much for lenovo
        )
        max_samples = 10_000

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
            25_000,  # seems to be minimum for valid config
        )

        # --- No Osci Streaming to LSL for AO testing - to free up CPU load
        #       which is needed by the AO streaming app
        # lsloutlet = get_stream_outlet(
        #     stream_name=STREAM_NAME, sfreq=TARGET_SRATE_HZ, n_channels=3
        # )
        #
        # pget_data = partial(get_data, lsloutlet=lsloutlet)
        # cback = CALLBACK(pget_data)
        # pupdate = partial(update, device=device, callback=cback)
        #
        while True and device.handle is not None and not stop_event.is_set():
            # fetch data from the osci
            CURRIDX = 0
            LSL_LAST_PUSH = time.perf_counter_ns()
            sleep_s(1)
            # pupdate()


def get_main_thread() -> tuple[threading.Thread, threading.Event]:
    stop_event = threading.Event()
    stop_event.clear()

    thread = threading.Thread(target=main, kwargs={"stop_event": stop_event})
    thread.start()

    return thread, stop_event


if __name__ == "__main__":
    Fire(main)
