import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter
import gpiozero  # Recommended library for Pi 5
import time
import tkinter as tk
import threading

# --- System settings ---
# 384 kHz - sample rate of dodotronic ultramic384K_evo microphone
FS = 384000
CHUNK_SIZE = 2048
WINDOW_SIZE = 8192
THRESHOLD_RMS = 0.02
PEAK_RATIO = 10.0    # "Needle" ratio (peak must be 10x above average)
LOW_CUT = 50000
HIGH_CUT = 90000
GPIO_PIN = 18        # BCM pin number

def find_dodotronic_device():
    """Search for dodotronic microphone and return device index, or None"""
    devices = sd.query_devices()
    keywords = ['dodotronic', 'ultramic', '384k', '384', 'evo']
    for i, device in enumerate(devices):
        if device['max_input_channels'] <= 0:
            continue
        name_lower = device['name'].lower()
        for kw in keywords:
            if kw in name_lower:
                try:
                    sd.check_input_settings(device=i, samplerate=FS, channels=1)
                    return i
                except Exception:
                    pass
    return None

# --- GPIO setup for Raspberry Pi 5 ---
# gpiozero works well on Pi 5 and manages the new kernel daemons
ttl_out = gpiozero.OutputDevice(GPIO_PIN)

# --- Band-pass filter ---
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

B, A = butter_bandpass(LOW_CUT, HIGH_CUT, FS)

# Global buffer and detection state (shared with GUI)
data_buffer = np.zeros(WINDOW_SIZE)
is_detected = False
detection_status = {'value': False}  # for GUI polling

def verify_peak_with_fft(signal):
    """
    Run FFT and check for a prominent spectral "needle" in the USV band.
    """
    # FFT (real FFT is more efficient)
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1/FS)
    
    # Restrict to relevant frequency range (50-90 kHz)
    mask = (freqs >= LOW_CUT) & (freqs <= HIGH_CUT)
    if not np.any(mask): 
        return False, 0
    
    relevant_mags = fft_vals[mask]
    peak_val = np.max(relevant_mags)
    
    # Average magnitude across full spectrum (noise floor)
    avg_noise = np.mean(fft_vals)
    
    # Check "needle" ratio - is the USV-band peak well above the noise?
    if peak_val > (avg_noise * PEAK_RATIO):
        peak_freq = freqs[mask][np.argmax(relevant_mags)]
        return True, peak_freq
    
    return False, 0

def process_audio(indata, frames, time_info, status):
    global data_buffer, is_detected
    
    # Update buffer (sliding window)
    data_buffer = np.roll(data_buffer, -frames)
    data_buffer[-frames:] = indata[:, 0]
    
    # Filter the windowed data
    filtered_data = lfilter(B, A, data_buffer)
    
    # Step 1: Quick RMS check (saves processing time)
    rms = np.sqrt(np.mean(filtered_data**2))
    
    if rms > THRESHOLD_RMS:
        # Step 2: Verify "needle" with FFT
        success, freq = verify_peak_with_fft(filtered_data)
        
        if success:
            if not is_detected:
                ttl_out.on()  # TTL on
                print(f"USV CONFIRMED! Freq: {freq/1000:.1f} kHz | RMS: {rms:.4f}")
                is_detected = True
        else:
            # Strong noise but not a "needle" (e.g. door slam)
            if is_detected:
                ttl_out.off()
                is_detected = False
    else:
        # Back to quiet
        if is_detected:
            ttl_out.off()
            is_detected = False
    
    detection_status['value'] = is_detected

# --- GUI and run ---
device_index = find_dodotronic_device()
if device_index is not None:
    dev_name = sd.query_devices(device_index)['name']
else:
    dev_name = "Default device"

stream_kwargs = dict(samplerate=FS, channels=1, callback=process_audio, blocksize=CHUNK_SIZE)
if device_index is not None:
    stream_kwargs['device'] = device_index

running = False
stream = None
listener_thread = None

def run_listener():
    global stream, running
    try:
        with sd.InputStream(**stream_kwargs) as stream:
            while running:
                time.sleep(0.1)
    except Exception as e:
        print("Listener error:", e)
    finally:
        if running is False:
            try:
                ttl_out.off()
            except Exception:
                pass

def toggle_run():
    global running, stream, listener_thread
    if running:
        running = False
        if listener_thread is not None:
            listener_thread.join(timeout=2.0)
        btn.config(text="Start")
        status_label.config(text="Stopped", bg="gray")
    else:
        running = True
        btn.config(text="Stop")
        listener_thread = threading.Thread(target=run_listener, daemon=True)
        listener_thread.start()

def poll_status():
    if running:
        if detection_status['value']:
            status_label.config(text="USV detected", bg="green")
        else:
            status_label.config(text="No USV", bg="gray")
    root.after(150, poll_status)

root = tk.Tk()
root.title("Online USV Detector")
root.geometry("320x140")
root.resizable(False, False)

mic_text = dev_name if len(dev_name) <= 32 else dev_name[:29] + "..."
tk.Label(root, text=f"Mic: {mic_text}", font=("Arial", 9)).pack(pady=(8, 2))
btn = tk.Button(root, text="Start", command=toggle_run, width=12, height=2, font=("Arial", 12))
btn.pack(pady=10)

tk.Label(root, text="Status:", font=("Arial", 10)).pack(anchor=tk.W, padx=10)
status_frame = tk.Frame(root)
status_frame.pack(fill=tk.X, padx=10, pady=(2, 10))
status_label = tk.Label(status_frame, text="Stopped", font=("Arial", 14, "bold"), bg="gray",
                       width=14, relief=tk.RIDGE, padx=8, pady=4)
status_label.pack()

poll_status()

def on_closing():
    global running
    running = False
    try:
        ttl_out.off()
        ttl_out.close()
    except Exception:
        pass
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()