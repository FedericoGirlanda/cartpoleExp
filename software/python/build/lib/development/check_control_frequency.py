import numpy as np
from quanser.hardware import HIL, HILError
import time

channels = np.array([0, 1], dtype=np.uint32)
num_channels = len(channels)
buffer_in = np.zeros(num_channels, dtype=np.float64)
buffer_out = np.array([5.0, 0.0], dtype=np.float64)
card = HIL()

a = None

dt = []
try:
    card.open("q2_usb", "0")
    time_0 = time.time()
    for k in range(10000):
        print(buffer_in)
        time_start = time.time() - time_0
        card.read_analog(channels, num_channels, buffer_in)

        buffer_out[0] = np.sin(time_start*np.pi * 2.0 * 4) * 2.0
        card.write_analog(channels, num_channels, buffer_out)

        time_after_read_write_cycle = time.time() - time_0
        dt.append(time_after_read_write_cycle - time_start)

    buffer_out = np.array([0.0, 0.0], dtype=np.float64)
    card.write_analog(channels, num_channels, buffer_out)

    if card.is_valid():
        card.close()
except HILError as e:
    print(e.get_error_message())
except KeyboardInterrupt:
    print("Aborted")
finally:
    if card.is_valid():
        card.close()

print(f'control frequency is {np.mean(1/np.array(dt)) / 1000} plus/minus {np.std(1/np.array(dt)) / 1000} kHz')

print('all done, closing')
