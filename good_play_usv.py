# import sounddevice as sd
# import soundfile as sf
# 
# # Load the WAV file
# data, fs = sf.read("two_kicks.wav")
# 
# print(f"File sample rate: {fs} Hz")
# 
# # Check available devices
# devices = sd.query_devices()
# print("\nAvailable devices:")
# for i, d in enumerate(devices):
#     print(f"{i}: {d['name']}")
# 
# # Choose your Scarlett output device (replace index if needed)
# # Find the index where the name includes "Scarlett"
# device_index = next(i for i, d in enumerate(devices) if "Scarlett" in d["name"])
# 
# print(f"\nUsing output device index: {device_index}")
# 
# # Play continuously at the original sample rate (e.g. 192000 Hz)
# while True:
#     sd.play(data, fs, device=device_index)
#     sd.wait()
#

import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.signal import resample

# Load the file
filename = "pup ICR USVs.wav"
data, fs = sf.read(filename)
print(f"Original sample rate: {fs} Hz")

# Target sample rate (Scarlett limit)
target_fs = 192000

# Resample if needed
if fs > target_fs:
    factor = target_fs / fs
    num_samples = int(len(data) * factor)
    print(f"Resampling from {fs} Hz to {target_fs} Hz ({factor:.2f}x slower)")
    data = resample(data, num_samples)
    fs = target_fs

# Detect Scarlett device
devices = sd.query_devices()
device_index = next(i for i, d in enumerate(devices) if "Scarlett" in d["name"])
print(f"Using output device: {devices[device_index]['name']}")

# Infinite playback loop
while True:
    sd.play(data, fs, device=device_index)
    sd.wait()

