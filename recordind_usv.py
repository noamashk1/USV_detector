import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

fs = 192000
duration = 60
print("Recording..")
data = sd.rec(int(fs*duration),samplerate = fs, channels =1, dtype ='float32')
sd.wait()
print("Done")

wav.write("test_ultrasonic_gain_high.wav", fs, data)