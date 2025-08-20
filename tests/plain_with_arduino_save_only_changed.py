# --> Here I am only storing values if there is a significant change happening
# --> instead of osci buffer values, I store time stamps
#
import time
from ctypes import POINTER, c_int16, c_uint32
from functools import partial

import numpy as np
import pandas as pd
import plotly.express as px
import serial
from fire import Fire
from picosdk.ctypes_wrapper import C_CALLBACK_FUNCTION_FACTORY
from picosdk.device import Device
from picosdk.functions import assert_pico2000_ok
from picosdk.PicoDeviceEnums import picoEnum
from picosdk.ps2000 import (
    ps2000 as ps,
)  # although my osci is a 2204a, it requires the older ps2000 SDK - see here: https://www.picotech.com/support/viewtopic.php?t=13171

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

TRIGGER_BUFFER = [(0, 0)]
B_POWER = [(0, 0)]


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

    global TRIGGER_BUFFER, B_POWER

    now = time.time_ns()
    # vc = max(min(a[-1], 15_000), 0)
    if a[-1] > 15_000 and TRIGGER_BUFFER[-1][1] == 0:
        arduino.write(b"u")
        arduino.write(b"d")
        TRIGGER_BUFFER.append((now, 1))
    elif a[-1] < 15_000 and TRIGGER_BUFFER[-1][1] == 1:
        TRIGGER_BUFFER.append((now, 0))

    if b[-1] > 5000 and B_POWER[-1][1] == 0:
        B_POWER.append((now, 1))
    elif b[-1] < 5000 and B_POWER[-1][1] == 1:
        B_POWER.append((now, 0))


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


def update(device: Device, callback: None = None, arduino: serial.Serial | None = None):
    ps.ps2000_get_streaming_last_values(device.handle, callback)


def main(t_test_s: float = 10):
    t_test_s = 20

    global TRIGGER_BUFFER, B_POWER
    TRIGGER_BUFFER, B_POWER = [(0, 0)], [(0, 0)]
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
        sample_interval = 300  # sample interval in ns
        max_samples = 100_000

        res = ps.ps2000_run_streaming_ns(
            device.handle,
            sample_interval,
            2,
            max_samples,
            False,
            agg_factor,
            1_000_000,
        )
        time.sleep(2)

        with serial.Serial(port="COM3", baudrate=19200, timeout=0.1) as arduino:
            pget_data = partial(get_data, arduino=arduino)
            cback = CALLBACK(pget_data)
            pupdate = partial(update, device=device, callback=cback, arduino=arduino)

            tstart = time.time_ns()

            while (
                device.handle is not None and time.time_ns() - tstart < t_test_s * 10**9
            ):
                # fetch data from the osci
                pupdate()


def evaluate_time_to_channel_b_rise():
    global TRIGGER_BUFFER, B_POWER
    tarr = np.array(TRIGGER_BUFFER)[1:]
    barr = np.array(B_POWER)[1:]

    # make all relative to first
    tmin = min(tarr[:, 0].min(), barr[:, 0].min())
    tarr[:, 0] -= tmin
    barr[:, 0] -= tmin
    # tarr[:, 0] /= 10**9
    # barr[:, 0] /= 10**9

    # get only increases
    tinc = tarr[tarr[:, 1] == 1]
    binc = barr[barr[:, 1] == 1]

    # sometimes a bpower reaction is scipped -> less values in binc.
    # align them by always choosing the next closest within tinc
    ix = [tinc[tinc <= t].shape[0] for t in binc]

    df = pd.DataFrame(
        {
            "t_s": np.hstack([tinc_s[:, 0], binc[:, 0]]) / 10**9,
            "y": [0] * tinc_s.shape[0] + [1] * binc.shape[0],
            "type": ["trigger"] * tinc_s.shape[0] + ["bpower"] * binc.shape[0],
        }
    )
    fig2 = px.scatter(df, x="t_s", y="y", color="type")
    fig2.show()


if __name__ == "__main__":
    fig = Fire(main)
