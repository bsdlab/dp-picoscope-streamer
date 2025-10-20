import logging
import threading
import time

import numpy as np

from picoscope_streamer.main import get_stream_outlet

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(ch)

logger.setLevel(logging.DEBUG)


def start_saw_stream(
    push_freq: int = 1000,
    stim_freq: int = 20,
    stop_event: threading.Event = threading.Event(),
):
    outlet = get_stream_outlet(stream_name="SOURCE", sfreq=push_freq, n_channels=1)

    a = np.linspace(0, 65200, int((push_freq / stim_freq) // 2))
    basic_shape = np.hstack([a, a[::-1]]) - 32600
    le = basic_shape.shape[0]
    i = 0
    tlast = time.time_ns()

    t_to_s = 1 / 10**9 * push_freq

    while not stop_event.is_set():
        dt = time.time_ns() - tlast
        req_samples = round(
            dt * t_to_s
        )  # round will result in more accurate timings overall
        # logger.debug(f">>> {req_samples=}, {dt=}")
        if req_samples > 0:
            tlast = time.time_ns()
            for _ in range(req_samples):
                outlet.push_sample([basic_shape[i % le]])
                i += 1

    print(f">>> Stopping stream - {stop_event.is_set()=}")


if __name__ == "__main__":
    stopev = threading.Event()
    try:
        start_saw_stream(stop_event=stopev)

    except KeyboardInterrupt as err:
        stopev.set()
        raise err
