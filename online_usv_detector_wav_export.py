import numpy as np
import sounddevice as sd
import time
import wave
import queue
import csv
from datetime import datetime

try:
    import lgpio
except ModuleNotFoundError:
    lgpio = None

import os

import tkinter as tk
from tkinter import messagebox, filedialog
import threading

# --- System settings ---
# 384 kHz - sample rate of dodotronic ultramic384K_evo microphone
FS = 384000
CHUNK_SIZE = 2048
WINDOW_SIZE = 8192

# Same detection as USV_recorder_analyzer: FFT on raw window, RMS of |FFT|^2 in ultrasonic band
THRESHOLD_RMS = 0.6   # default matches analyzer "Ultrasonic Threshold (RMS)"
LOW_CUT = 50000
HIGH_CUT = 90000
GPIO_PIN = 18        # BCM pin for TTL (set from UI at start)


class LgpioTTL:
    """TTL output via lgpio (same interface as gpiozero.OutputDevice)."""

    def __init__(self, gpio_pin):
        self._h = lgpio.gpiochip_open(0)
        err = lgpio.gpio_claim_output(self._h, gpio_pin)
        if err < 0:
            raise RuntimeError(f"lgpio gpio_claim_output failed: {err}")
        self._pin = gpio_pin

    def on(self):
        lgpio.gpio_write(self._h, self._pin, 1)

    def off(self):
        lgpio.gpio_write(self._h, self._pin, 0)

    def close(self):
        try:
            lgpio.gpio_free(self._h, self._pin)
        except Exception:
            pass
        lgpio.gpiochip_close(self._h)


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


# --- GPIO (Raspberry Pi), created on Start from UI pin ---
ttl_out = None

# Runtime state
data_buffer = np.zeros(WINDOW_SIZE)
is_detected = False
detection_status = {'value': False}

# Precise timing counters (for CSV log and later spectrogram markers)
samples_processed = 0
last_chunk_end_time_s = 0.0
last_rms_value = 0.0

# CSV log writer runtime state
log_queue = None
log_writer_thread = None
log_path = None
log_dropped_rows = 0

event_lock = threading.Lock()
current_event = None  # dict with fields until it ends

# WAV writer runtime state (queue decouples callback from disk IO)
recording_queue = None
writer_thread = None
recording_path = None
dropped_chunks = 0


def compute_band_rms_and_peak(window, low_hz, high_hz, fs):
    """
    Same logic as USV_recorder_analyzer: FFT on window, then RMS of |FFT|^2 in [low_hz, high_hz].
    Returns (rms, peak_freq_hz). peak_freq_hz is 0 if band is empty.
    """
    fft = np.fft.rfft(window)
    freqs = np.fft.rfftfreq(len(window), 1 / fs)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    ultrasonic_fft = fft[mask]
    if len(ultrasonic_fft) == 0:
        return 0.0, 0.0
    rms = np.sqrt(np.mean(np.abs(ultrasonic_fft) ** 2))
    peak_idx = np.argmax(np.abs(ultrasonic_fft))
    peak_freq = freqs[mask][peak_idx]
    return rms, peak_freq


def wav_writer_worker(wav_path, audio_queue, fs):
    """Write raw PCM16 mono frames into a single WAV file until sentinel."""
    global dropped_chunks
    try:
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # PCM16
            wf.setframerate(fs)

            while True:
                chunk = audio_queue.get()
                if chunk is None:
                    break
                wf.writeframesraw(chunk)
    except Exception as e:
        # Don't crash the whole GUI; make it visible in console.
        print(f"wav_writer_worker error: {e}")
    finally:
        # When disk write fails, it's still useful to know how much was dropped upstream.
        if dropped_chunks > 0:
            print(f"WAV writer: dropped chunks while writing: {dropped_chunks}")


def log_writer_worker(csv_path, q):
    """Write detection intervals (CSV) until sentinel None is received."""
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "start_time_s",
                "end_time_s",
                "duration_s",
                "sampling_rate_hz",
                "low_cut_hz",
                "high_cut_hz",
                "threshold_rms",
                "rms_start",
                "rms_end",
                "peak_freq_hz",
            ])
            while True:
                row = q.get()
                if row is None:
                    break
                writer.writerow(row)
                f.flush()
    except Exception as e:
        print(f"log_writer_worker error: {e}")


def process_audio(indata, frames, time_info, status):
    global data_buffer, is_detected, dropped_chunks, recording_queue
    global samples_processed, last_chunk_end_time_s, last_rms_value
    global log_queue, log_dropped_rows, current_event

    # Timing for this callback chunk.
    # We timestamp detection transitions at the end of the chunk we just appended into the sliding window.
    chunk_end_time_s = (samples_processed + frames) / FS

    # Sliding window (raw audio, like analyzer)
    data_buffer = np.roll(data_buffer, -frames)
    data_buffer[-frames:] = indata[:, 0]

    # Same criterion as USV_recorder_analyzer: FFT-band RMS > threshold
    rms, peak_freq = compute_band_rms_and_peak(data_buffer, LOW_CUT, HIGH_CUT, FS)
    last_rms_value = rms

    if rms > THRESHOLD_RMS:
        if not is_detected:
            if ttl_out is not None:
                ttl_out.on()
            print(f"USV CONFIRMED! Freq: {peak_freq/1000:.1f} kHz | RMS: {rms:.4f}")

            # Start an interval event for CSV logging.
            with event_lock:
                current_event = {
                    "start_time_s": max(0.0, chunk_end_time_s - (WINDOW_SIZE / FS)),
                    "low_cut_hz": int(LOW_CUT),
                    "high_cut_hz": int(HIGH_CUT),
                    "threshold_rms": float(THRESHOLD_RMS),
                    "rms_start": float(rms),
                    "rms_end": None,
                    "peak_freq_hz": float(peak_freq),
                }
            is_detected = True
    else:
        if is_detected:
            if ttl_out is not None:
                ttl_out.off()

            # Close and enqueue the interval event.
            with event_lock:
                if current_event is not None:
                    current_event["end_time_s"] = chunk_end_time_s
                    current_event["rms_end"] = float(rms)
                    duration_s = current_event["end_time_s"] - current_event["start_time_s"]

                    row = [
                        current_event["start_time_s"],
                        current_event["end_time_s"],
                        duration_s,
                        FS,
                        current_event["low_cut_hz"],
                        current_event["high_cut_hz"],
                        current_event["threshold_rms"],
                        current_event["rms_start"],
                        current_event["rms_end"],
                        current_event["peak_freq_hz"],
                    ]
                    try:
                        if log_queue is not None:
                            log_queue.put_nowait(row)
                    except queue.Full:
                        log_dropped_rows += 1
                    current_event = None

            is_detected = False

    detection_status['value'] = is_detected

    # Enqueue audio for WAV saving (callback should never block)
    if recording_queue is not None:
        try:
            audio = np.clip(indata[:, 0], -1.0, 1.0)
            pcm16 = (audio * 32767).astype(np.int16)
            recording_queue.put_nowait(pcm16.tobytes())
        except queue.Full:
            dropped_chunks += 1

    # Update timing counters after all processing for this chunk.
    samples_processed += frames
    last_chunk_end_time_s = chunk_end_time_s


# --- GUI and run ---
device_index = find_dodotronic_device()
if device_index is not None:
    dev_name = sd.query_devices(device_index)['name']
else:
    dev_name = "Default device"

running = False
listener_thread = None
_stream_kwargs = None


def run_listener():
    global running
    if _stream_kwargs is None:
        return
    try:
        with sd.InputStream(**_stream_kwargs) as stream:
            while running:
                time.sleep(0.1)
    except Exception as e:
        print("Listener error:", e)
    finally:
        # TTL cleanup is handled here (WAV cleanup is handled by stop logic).
        if running is False and ttl_out is not None:
            try:
                ttl_out.off()
            except Exception:
                pass


def apply_params_and_start():
    """Read GUI params, validate, set globals, build stream_kwargs."""
    global data_buffer, THRESHOLD_RMS, LOW_CUT, HIGH_CUT, CHUNK_SIZE, WINDOW_SIZE, GPIO_PIN, ttl_out, _stream_kwargs
    try:
        low_khz = float(low_cut_var.get())
        high_khz = float(high_cut_var.get())
        low = int(low_khz * 1000)
        high = int(high_khz * 1000)
        rms = float(rms_var.get())
        chunk_ms = float(chunk_ms_var.get())
        window_ms = float(window_ms_var.get())
        gpio_pin = int(gpio_pin_var.get())
    except ValueError:
        return False

    if low <= 0 or high <= low or rms <= 0 or chunk_ms <= 0 or window_ms < chunk_ms or gpio_pin < 0:
        return False

    CHUNK_SIZE = int(FS * chunk_ms / 1000.0)
    WINDOW_SIZE = int(FS * window_ms / 1000.0)
    if CHUNK_SIZE <= 0 or WINDOW_SIZE < CHUNK_SIZE:
        return False

    LOW_CUT = low
    HIGH_CUT = high
    THRESHOLD_RMS = rms
    GPIO_PIN = gpio_pin

    if ttl_out is not None:
        try:
            ttl_out.close()
        except Exception:
            pass

    try:
        ttl_out = LgpioTTL(GPIO_PIN) if lgpio else None
    except Exception:
        ttl_out = None

    data_buffer = np.zeros(WINDOW_SIZE)
    _stream_kwargs = dict(samplerate=FS, channels=1, callback=process_audio, blocksize=CHUNK_SIZE)
    if device_index is not None:
        _stream_kwargs['device'] = device_index
    return True


def stop_recording_and_stream():
    """Stop the audio stream and finalize WAV file (if enabled)."""
    global running, listener_thread, recording_queue, writer_thread, recording_path, dropped_chunks
    global log_queue, log_writer_thread, log_path, log_dropped_rows
    global current_event, last_chunk_end_time_s, last_rms_value
    running = False
    if listener_thread is not None:
        listener_thread.join(timeout=2.0)

    # Tell writer thread to finish.
    if recording_queue is not None:
        try:
            recording_queue.put(None, timeout=2.0)
        except Exception:
            pass

    if writer_thread is not None:
        writer_thread.join(timeout=5.0)

    if recording_path:
        print(f"Session folder: {os.path.dirname(recording_path)}")
        print(f"Recording saved to: {recording_path}")
        if dropped_chunks > 0:
            print(f"Dropped WAV chunks: {dropped_chunks}")

    # If we stopped while still detected, close and log the last interval.
    with event_lock:
        if current_event is not None:
            current_event["end_time_s"] = float(last_chunk_end_time_s)
            current_event["rms_end"] = float(last_rms_value)
            duration_s = current_event["end_time_s"] - current_event["start_time_s"]

            row = [
                current_event["start_time_s"],
                current_event["end_time_s"],
                duration_s,
                FS,
                current_event["low_cut_hz"],
                current_event["high_cut_hz"],
                current_event["threshold_rms"],
                current_event["rms_start"],
                current_event["rms_end"],
                current_event["peak_freq_hz"],
            ]
            try:
                if log_queue is not None:
                    log_queue.put_nowait(row)
            except queue.Full:
                log_dropped_rows += 1
            current_event = None

    if log_queue is not None:
        try:
            log_queue.put(None, timeout=2.0)
        except Exception:
            pass

    if log_writer_thread is not None:
        log_writer_thread.join(timeout=5.0)

    if log_path:
        print(f"Detection log saved to: {log_path}")
        if log_dropped_rows > 0:
            print(f"Dropped CSV rows: {log_dropped_rows}")

    recording_queue = None
    writer_thread = None
    recording_path = None
    dropped_chunks = 0

    log_queue = None
    log_writer_thread = None
    log_path = None
    log_dropped_rows = 0


def toggle_run():
    global running, listener_thread, recording_queue, writer_thread, recording_path, dropped_chunks
    global log_queue, log_writer_thread, log_path, log_dropped_rows
    global samples_processed, last_chunk_end_time_s, last_rms_value, current_event
    if running:
        stop_recording_and_stream()
        btn.config(text="Start")
        status_label.config(text="Stopped", bg="gray")
        return

    if not apply_params_and_start():
        messagebox.showerror("Invalid parameters", "Check: low < high, all positive, window (ms) >= chunk (ms), GPIO pin >= 0.")
        return

    # Choose session name: a folder with this name is created; WAV + log go inside it.
    default_name = f"usv_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    file_path = filedialog.asksaveasfilename(
        title="Save session (folder name = file name without .wav)",
        defaultextension=".wav",
        initialfile=default_name,
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
    )
    if not file_path:
        # User canceled; do not start streaming.
        btn.config(text="Start")
        status_label.config(text="Stopped", bg="gray")
        return

    if not file_path.lower().endswith(".wav"):
        file_path += ".wav"

    parent_dir = os.path.dirname(os.path.abspath(file_path))
    session_basename = os.path.splitext(os.path.basename(file_path))[0]
    if not session_basename.strip():
        messagebox.showerror("Invalid name", "Please choose a non-empty file name.")
        btn.config(text="Start")
        status_label.config(text="Stopped", bg="gray")
        return

    session_dir = os.path.join(parent_dir, session_basename)
    try:
        os.makedirs(session_dir, exist_ok=True)
    except OSError as e:
        messagebox.showerror("Folder error", f"Could not create session folder:\n{e}")
        btn.config(text="Start")
        status_label.config(text="Stopped", bg="gray")
        return

    recording_path = os.path.join(session_dir, f"{session_basename}.wav")
    log_path = os.path.join(session_dir, f"{session_basename}.log.csv")
    dropped_chunks = 0

    # Reset timing and event state for this run.
    samples_processed = 0
    last_chunk_end_time_s = 0.0
    last_rms_value = 0.0
    current_event = None
    log_dropped_rows = 0

    recording_queue = queue.Queue(maxsize=1000)  # stores PCM16 bytes chunks
    log_queue = queue.Queue(maxsize=1000)
    log_writer_thread = threading.Thread(
        target=log_writer_worker,
        args=(log_path, log_queue),
        daemon=True,
    )
    log_writer_thread.start()

    writer_thread = threading.Thread(
        target=wav_writer_worker,
        args=(recording_path, recording_queue, FS),
        daemon=True
    )
    writer_thread.start()

    running = True
    btn.config(text="Stop")
    status_label.config(text="Recording...", bg="green")
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
root.title("Online USV Detector (WAV Export)")
root.geometry("420x380")
root.resizable(False, False)

mic_text = dev_name if len(dev_name) <= 32 else dev_name[:29] + "..."
tk.Label(root, text=f"Mic: {mic_text}", font=("Arial", 9)).pack(pady=(6, 2))

# Parameters frame
params_frame = tk.LabelFrame(root, text="Parameters", padx=8, pady=6)
params_frame.pack(fill=tk.X, padx=8, pady=4)

low_cut_var = tk.StringVar(value="50")
high_cut_var = tk.StringVar(value="90")
rms_var = tk.StringVar(value="0.6")
chunk_ms_var = tk.StringVar(value="50")
window_ms_var = tk.StringVar(value="100")
gpio_pin_var = tk.StringVar(value="18")

row = 0
tk.Label(params_frame, text="Low cut (kHz):", width=14, anchor=tk.W).grid(row=row, column=0, sticky=tk.W, pady=2)
tk.Entry(params_frame, textvariable=low_cut_var, width=10).grid(row=row, column=1, pady=2)
row += 1
tk.Label(params_frame, text="High cut (kHz):", width=14, anchor=tk.W).grid(row=row, column=0, sticky=tk.W, pady=2)
tk.Entry(params_frame, textvariable=high_cut_var, width=10).grid(row=row, column=1, pady=2)
row += 1
tk.Label(params_frame, text="Threshold (FFT RMS):", width=22, anchor=tk.W).grid(row=row, column=0, sticky=tk.W, pady=2)
tk.Entry(params_frame, textvariable=rms_var, width=10).grid(row=row, column=1, pady=2)
row += 1
tk.Label(params_frame, text="Chunk (ms):", width=14, anchor=tk.W).grid(row=row, column=0, sticky=tk.W, pady=2)
tk.Entry(params_frame, textvariable=chunk_ms_var, width=10).grid(row=row, column=1, pady=2)
row += 1
tk.Label(params_frame, text="Window (ms):", width=14, anchor=tk.W).grid(row=row, column=0, sticky=tk.W, pady=2)
tk.Entry(params_frame, textvariable=window_ms_var, width=10).grid(row=row, column=1, pady=2)
row += 1
tk.Label(params_frame, text="GPIO pin (BCM):", width=14, anchor=tk.W).grid(row=row, column=0, sticky=tk.W, pady=2)
tk.Entry(params_frame, textvariable=gpio_pin_var, width=10).grid(row=row, column=1, pady=2)

btn = tk.Button(root, text="Start", command=toggle_run, width=12, height=2, font=("Arial", 12))
btn.pack(pady=8)

tk.Label(root, text="Status:", font=("Arial", 10)).pack(anchor=tk.W, padx=10)
status_frame = tk.Frame(root)
status_frame.pack(fill=tk.X, padx=10, pady=(2, 10))
status_label = tk.Label(
    status_frame,
    text="Stopped",
    font=("Arial", 14, "bold"),
    bg="gray",
    width=18,
    relief=tk.RIDGE,
    padx=8,
    pady=4
)
status_label.pack()

poll_status()


def on_closing():
    global running
    try:
        if running:
            stop_recording_and_stream()
    except Exception:
        pass

    if ttl_out is not None:
        try:
            ttl_out.off()
            ttl_out.close()
        except Exception:
            pass

    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
