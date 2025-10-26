import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal

# Load the loop audio file
print("Loading audio file...")
file_fs, loop_audio_data = wav.read("pup ICR USVs.wav")

print(f"Original: {file_fs} Hz, {len(loop_audio_data)/file_fs:.2f}s")

# Convert to float32
if loop_audio_data.dtype == np.int16:
    loop_audio_data = loop_audio_data.astype(np.float32) / 32767.0
elif loop_audio_data.dtype == np.int32:
    loop_audio_data = loop_audio_data.astype(np.float32) / 2147483647.0

# Resample to 192000 Hz
target_fs = 192000
if file_fs != target_fs:
    num_samples = int(len(loop_audio_data) * target_fs / file_fs)
    loop_audio_data = signal.resample(loop_audio_data, num_samples)
    print(f"Resampled to {target_fs} Hz")

print(f"Now: {len(loop_audio_data)/target_fs:.2f}s")
print("Playing audio...")

# Try to play
sd.play(loop_audio_data, samplerate=target_fs)
sd.wait()

print("Done!")

