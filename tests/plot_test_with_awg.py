# testing different parameters for smoothes possible representation
from ctypes import POINTER, c_int16, c_uint32
from functools import partial
from picoscope_streamer.awg import set_sig_gen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from picosdk.ps2000 import ps2000
from picosdk.functions import assert_pico2000_ok
from picosdk.PicoDeviceEnums import picoEnum
from picosdk.ctypes_wrapper import C_CALLBACK_FUNCTION_FACTORY

import time

CALLBACK = C_CALLBACK_FUNCTION_FACTORY(
    None,
    POINTER(POINTER(c_int16)),
    c_int16,
    c_uint32,
    c_int16,
    c_int16,
    c_uint32,
)

# just making it large enough
adc_values = np.zeros(200_000_000)
control_points = np.zeros((20_000_000, 2))

nvalues = []
times = []

lvals = []

ICURR = 0
ICONT = 0


def get_overview_buffers(
    buffers, _overflow, _triggered_at, _triggered, _auto_stop, n_values
):
    global ICURR, adc_values, nvalues, control_points, ICONT
    vals = buffers[0][0:n_values]
    adc_values[ICURR : ICURR + n_values] = vals
    ICURR += n_values
    control_points[ICONT, :] = [ICURR, vals[0]]
    ICONT += 1
    nvalues.append(n_values)


def adc_to_mv(values, range_, bitness=16):
    v_ranges = [10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000]

    return [(x * v_ranges[range_]) / (2 ** (bitness - 1) - 1) for x in values]


with ps2000.open_unit() as device:
    print("Device info: {}".format(device.info))

    res = ps2000.ps2000_set_channel(
        device.handle,
        picoEnum.PICO_CHANNEL["PICO_CHANNEL_A"],
        True,
        picoEnum.PICO_COUPLING["PICO_DC"],
        ps2000.PS2000_VOLTAGE_RANGE["PS2000_200MV"],
    )
    assert_pico2000_ok(res)

    status = {}
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

    """ int16_t ps2000_run_streaming_ns
    (
        int16_t            handle,
        uint32_t           sample_interval,
        PS2000_TIME_UNITS  time_units,  2 == nanos
        uint32_t           max_samples,
        int16_t            auto_stop,
        uint32_t           noOfSamplesPerAggregate,
        uint32_t           overview_buffer_size
    );
        """
    res = ps2000.ps2000_run_streaming_ns(
        device.handle, 500, 2, 10_000, False, 5, 25_000
    )
    assert_pico2000_ok(res)

    start_time = time.time_ns()

    callback = CALLBACK(get_overview_buffers)
    while time.time_ns() - start_time < 5_000_000_000:
        ps2000.ps2000_get_streaming_last_values(device.handle, callback)

    end_time = time.time_ns()

    ps2000.ps2000_stop(device.handle)

    print("-" * 80)
    imax = np.where(adc_values != 0)[0][-1]
    imax_c = np.where(control_points != 0)[0][-1]
    print(f"Collected: {imax}")
    print("-" * 80)

    mv_values = adc_to_mv(
        adc_values[:imax], ps2000.PS2000_VOLTAGE_RANGE["PS2000_50MV"]
    )
    # print(f"{min(nvalues)} - {max(nvalues)} - {len(adc_values)}")

    df = pd.DataFrame({"nvalues": nvalues})
    print(df.nvalues.value_counts())
    print(df.shape)

    fig, axs = plt.subplots(2, 1)

    axs[0].set_xlabel("time/ms")
    axs[0].set_ylabel("voltage/mV")
    time_grid = np.linspace(0, (end_time - start_time) * 1e-6, imax)
    axs[0].plot(
        time_grid,
        adc_values[:imax],
    )
    axs[0].plot(
        time_grid[control_points[:imax_c, 0].astype(int)],
        control_points[:imax_c, 1],
        "r*",
    )
    axs[1].plot(
        time_grid,
        mv_values[:imax],
    )

    plt.show()
