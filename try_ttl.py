# import lgpio
# import time

# GPIO = 18  # BCM numbering

# # Open GPIO chip (0 is correct for Raspberry Pi)
# h = lgpio.gpiochip_open(0)

# # Set GPIO18 as output
# lgpio.gpio_claim_output(h, GPIO)

# try:
#     while True:
#         lgpio.gpio_write(h, GPIO, 1)  # HIGH (3.3V)
#         time.sleep(0.5)               # pulse width
#         lgpio.gpio_write(h, GPIO, 0)  # LOW
#         time.sleep(2.5)               # rest of 3 sec period

# except KeyboardInterrupt:
#     pass

# finally:
#     lgpio.gpio_write(h, GPIO, 0)
#     lgpio.gpiochip_close(h)

import wave

file_path = "C:\\Users\\noam4\\Downloads\\file_example_WAV_1MG.wav"#"C:\\Users\\noam4\\Downloads\\pup ICR USVs.wav"

with wave.open(file_path, "rb") as wav_file:
    sampling_rate = wav_file.getframerate()
    print(f"Sampling Rate: {sampling_rate} Hz")