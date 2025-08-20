import ctypes
import time

from picosdk.functions import assert_pico2000_ok
from picosdk.ps2000 import ps2000 as ps


def set_sig_gen(
    status: dict = {},
    chandle: ctypes.c_short = ctypes.c_short(0),
    freq: float = 3,
    sig_type: int = 2,
    amp_uv: int = 2_000_000,
) -> dict:
    """Set the signal generator > should configure and start the AWG"""

    # wavetype = "PS2000_RAMPUP"
    wavetype = ctypes.c_int32(sig_type)
    # As of official docs
    # https://www.picotech.com/download/manuals/picoscope-2000-series-programmers-guide.pdf
    #   waveType values
    #   PS2000_SINE sine wave
    #   PS2000_SQUARE square wave
    #   PS2000_TRIANGLE triangle wave
    #   PS2000_RAMPUP rising sawtooth
    #   PS2000_RAMPDOWN falling sawtooth
    #   PS2000_DC_VOLTAGE DC voltage
    #   PS2000_GAUSSIAN Gaussian
    #   PS2000_SINC sin(x)/x
    #   PS2000_HALF_SINE half (full-wave rectified) sine#

    offset_voltage_uV = ctypes.c_int32(0)
    pk_to_pk_uV = ctypes.c_uint32(amp_uv)

    # For sweeps only
    startfreq = freq
    stopfreq = freq
    increment = 0
    dwelltime = 0  # time in seconds between changes in frequecy
    sweepType = ctypes.c_int32(1)
    sweeps = ctypes.c_uint32(0)

    # For checking types - this is the implementation:
    # ps2000.make_symbol("_set_sig_gen_arbitrary", "ps2000_set_sig_gen_arbitrary", c_int16,
    #              [c_int16, c_int32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_void_p, c_int32, c_int32,
    #               c_uint32], doc)
    status["SetSigGenBuiltIn"] = ps.ps2000_set_sig_gen_built_in(
        chandle,
        offset_voltage_uV,
        pk_to_pk_uV,
        wavetype,
        startfreq,
        stopfreq,
        increment,
        dwelltime,
        sweepType,
        sweeps,
    )
    assert_pico2000_ok(status["SetSigGenBuiltIn"])

    return status


if __name__ == "__main__":
    status = {}
    with ps.open_unit() as device:
        status = set_sig_gen(status=status, chandle=device.handle)
        time.sleep(20)
