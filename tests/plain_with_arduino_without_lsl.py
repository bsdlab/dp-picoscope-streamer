# For now this is a very simple module only made to work with a picoscope 2204A
import ctypes
import time
from ctypes import POINTER, c_int16, c_uint32
from functools import partial

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import serial
from fire import Fire
from picosdk.ctypes_wrapper import C_CALLBACK_FUNCTION_FACTORY
from picosdk.device import Device
from picosdk.functions import assert_pico2000_ok
from picosdk.PicoDeviceEnums import picoEnum
from picosdk.ps2000 import (
    ps2000 as ps,
)  # although my osci is a 2204a, it requires the older ps2000 SDK - see here: https://www.picotech.com/support/viewtopic.php?t=13171
from scipy.signal import find_peaks

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

DATA = []
CHUNK = 1


def get_data(
    buffers,
    _overflow,
    _triggered_at,
    _triggered,
    _auto_stop,
    n_values,
    arduino: serial.Serial | None = None,
    # mmapo: mmap.mmap | None = None,
):
    # From Docs: https://www.picotech.com/download/manuals/ps2000pg.en-10.pdf
    a = buffers[0][0:n_values]
    b = buffers[3][0:n_values]

    global DATA, CHUNK

    CHUNK *= -1

    # vc = max(min(a[-1], 15_000), 0)
    if a[-1] > 15_000:
        vc = 15000
        arduino.write(b"u")
        arduino.write(b"d")
    else:
        vc = 0

    for va, vb in zip(a, b):
        DATA.append((va, vb, vc, CHUNK * 10_000))

    # Alsways set to down after


def setup_osci(device: Device, channel_range_a: int = 3, channel_range_b: int = 3):
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


def get_timebasis(device: Device):
    timeInterval = ctypes.c_int32()
    timeUnits = ctypes.c_int32()
    oversample = ctypes.c_int16(1)
    maxSamplesReturn = ctypes.c_int32()

    timebase = 2
    time = ps.ps2000_get_timebase(
        device.handle,
        timebase,
        100_000,
        ctypes.byref(timeInterval),
        ctypes.byref(timeUnits),
        oversample,
        ctypes.byref(maxSamplesReturn),
    )

    return time


def update(device: Device, callback: None = None, arduino: serial.Serial | None = None):
    ps.ps2000_get_streaming_last_values(device.handle, callback)


def main(t_test_s: float = 10):
    t_test_s = 10

    global adc_samples, nnew, DATA
    DATA = []
    adc_samples = []
    ch_range_a = ps.PS2000_VOLTAGE_RANGE["PS2000_2V"]
    ch_range_b = ps.PS2000_VOLTAGE_RANGE["PS2000_2V"]

    with ps.open_unit() as device:
        status = setup_osci(
            device, channel_range_a=ch_range_a, channel_range_b=ch_range_b
        )

        # Activate the AWG
        freq_hz = 20.0  # we expect reaction to be faster than 10ms
        sig_type = 2  # Triangular is better for buffer stability (the edge in the saber tooth leads to problems)
        amp_uV = 2_000_000
        status = set_sig_gen(
            status,
            chandle=device.handle,
            freq=freq_hz,
            sig_type=sig_type,
            amp_uv=amp_uV,
        )

        agg_factor = 1
        sample_interval = 1000  # sample interval in ns
        max_samples = 10_000

        res = ps.ps2000_run_streaming_ns(
            device.handle,
            sample_interval,
            2,
            max_samples,
            False,
            agg_factor,
            15_000,
        )
        time.sleep(2)

        with serial.Serial(port="COM3", baudrate=19200, timeout=0.1) as arduino:
            pget_data = partial(get_data, arduino=arduino)
            cback = CALLBACK(pget_data)
            pupdate = partial(update, device=device, callback=cback, arduino=arduino)

            tstart = time.time_ns()
            tlast = time.time_ns()
            while device.handle is not None and tlast - tstart < 2 * 10**9:
                # fetch data from the osci
                pupdate()
                tlast = time.time_ns()
            act_pre = tlast - tstart

            DATA = []

            tstart = time.time_ns()
            tlast = time.time_ns()
            while device.handle is not None and tlast - tstart < t_test_s * 10**9:
                # fetch data from the osci
                pupdate()
                tlast = time.time_ns()

            act = tlast - tstart

    print(
        f"actual time: {act / 10**6:.4f}ms - effective srate {len(DATA) / (act / 10**9)}"
    )

    # Plot the results
    fig = go.Figure()
    nmax = 2_000_000
    trange = np.linspace(0, t_test_s, len(DATA))
    DATA = DATA[-nmax:]
    trange = trange[-nmax:]

    darr = np.array(DATA)

    fig = fig.add_trace(
        go.Scatter(
            x=trange,
            y=darr[:, 0],
            name="CH_A",
            mode="lines",
            line_color="#55f",
        )
    )
    fig = fig.add_trace(
        go.Scatter(
            x=trange,
            y=darr[:, 1],
            name="CH_B",
            mode="lines",
            line_color="#f55",
        )
    )
    fig = fig.add_trace(
        go.Scatter(
            x=trange,
            y=darr[:, 2],
            name="to_arduino",
            mode="lines",
            line_color="#333",
            opacity=0.5,
        )
    )
    fig = fig.add_trace(
        go.Scatter(
            x=trange,
            y=darr[:, 3],
            name="chunk",
            mode="lines",
            line_color="#3a3",
            opacity=0.5,
        )
    )
    fig = fig.update_xaxes(title="time [s]")

    fig.show()


def evaluate_time_to_channel_b_rise():
    global DATA
    darr = np.array(DATA)

    # Note scipy does not always take the first peak (visual inspection at max
    # sampling rate for the picoscope showed that 600 works well for 20Hz
    # signals -> the width of the area with peaks is about 200 * 250ns = 50us)
    apeaks = find_peaks(darr[:, 0], height=16312, distance=600)

    # get rise to left of peak
    rises = np.where(np.diff(darr[:, 2]) > 500)[0] + 1
    prepeak_rises = [rises[rises <= i][-1] for i in apeaks[0] if i >= rises[0]]

    # calc post peak reaction on channel B
    brises = np.where(np.diff(darr[:, 1]) > 5000)[0] + 1
    postpeak_rises = [brises[brises >= i][0] for i in apeaks[0] if i >= rises[0]]

    fig = go.Figure()
    fig = fig.add_scatter(y=darr[:, 0])
    fig = fig.add_scatter(y=darr[:, 2], line_color="#333", opacity=0.3)
    fig = fig.add_scatter(y=darr[:, 1], line_color="#f33", opacity=0.3)
    fig = fig.add_scatter(x=apeaks[0], y=apeaks[1]["peak_heights"], mode="markers")
    fig = fig.add_scatter(
        x=prepeak_rises,
        y=darr[prepeak_rises, 2],
        mode="markers",
        marker_color="#5f5",
    )
    fig = fig.add_scatter(
        x=postpeak_rises,
        y=darr[postpeak_rises, 1],
        mode="markers",
        marker_color="#2aa",
    )
    fig.show()

    # calc dts
    df = pd.DataFrame(
        {"ds": [post - pre for pre, post in zip(prepeak_rises, postpeak_rises)]}
    )
    df["dt"] = df.ds * (250 * 10**-9)

    fig2 = px.box(df, y="dt")
    fig2.show()


if __name__ == "__main__":
    fig = Fire(main)
