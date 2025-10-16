import tkinter as tk
from tkinter import messagebox
import threading
import sounddevice as sd
import numpy as np
import lgpio
import time

class UltrasonicDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title("Ultrasonic Detector")

        # GUI variables
        self.threshold_var = tk.DoubleVar(value=0.2)
        self.gpio_var = tk.IntVar(value=17)
        self.ttl_duration_var = tk.DoubleVar(value=0.05)
        self.min_freq_khz_var = tk.DoubleVar(value=30.0)
        self.running = False

        # GUI layout
        tk.Label(master, text="Ultrasonic Threshold (RMS):").pack(pady=5)
        tk.Entry(master, textvariable=self.threshold_var).pack(pady=2)

        tk.Label(master, text="Min Frequency (kHz):").pack(pady=5)
        tk.Entry(master, textvariable=self.min_freq_khz_var).pack(pady=2)

        tk.Label(master, text="GPIO Number:").pack(pady=5)
        tk.Entry(master, textvariable=self.gpio_var).pack(pady=2)

        tk.Label(master, text="TTL Duration (sec):").pack(pady=5)
        tk.Entry(master, textvariable=self.ttl_duration_var).pack(pady=2)

        self.start_button = tk.Button(master, text="Start", command=self.start, width=15)
        self.start_button.pack(pady=10)
        self.stop_button = tk.Button(master, text="Stop", command=self.stop, state=tk.DISABLED, width=15)
        self.stop_button.pack(pady=5)

        self.status_label = tk.Label(master, text="Status: Idle", font=("Arial", 10, "bold"))
        self.status_label.pack(pady=10)
        
        # הוספת אינדיקציות
        self.rms_label = tk.Label(master, text="RMS: 0.000", font=("Arial", 12))
        self.rms_label.pack(pady=5)
        
        self.detection_label = tk.Label(master, text="Detection: None", bg="gray", font=("Arial", 12, "bold"), width=20)
        self.detection_label.pack(pady=5)

        self.detector_thread = None
        self.handle = None

    #     def check_audio_devices(self):
#         """בדיקה שיש כרטיסי קול זמינים שתומכים בsample rate גבוה"""
#         try:
#             devices = sd.query_devices()
#             print("Available audio devices:")
#             for i, device in enumerate(devices):
#                 print(f"{i}: {device['name']} - Max input channels: {device['max_input_channels']}")
#             
#             # בדיקה שיש לפחות התקן אחד עם כניסות
#             input_devices = [d for d in devices if d['max_input_channels'] > 0]
#             if not input_devices:
#                 return False, "No input audio devices found"
#             
#             # בדיקה שהתקן ברירת המחדל תומך ב-192kHz
#             default_device = sd.query_devices(kind='input')
#             try:
#                 sd.check_input_settings(device=default_device['name'], 
#                                       samplerate=192000, 
#                                       channels=1)
#                 return True, f"Audio device ready: {default_device['name']}"
#             except Exception as e:
#                 return False, f"Default device doesn't support 192kHz: {str(e)}"
#                 
#         except Exception as e:
#             return False, f"Error checking audio devices: {str(e)}"

    def check_audio_devices(self):
        """בדיקה שיש כרטיסי קול זמינים שתומכים בsample rate גבוה"""
        try:
            devices = sd.query_devices()
            print("Available audio devices:")
            for i, device in enumerate(devices):
                print(f"{i}: {device['name']} - Max input channels: {device['max_input_channels']}")
            
            # חיפוש ספציפי אחר Scarlett 2i2
            scarlett_device = None
            for device in devices:
                device_name = device['name'].lower()
                if ('scarlett' in device_name or 'focusrite' in device_name) and device['max_input_channels'] > 0:
                    scarlett_device = device
                    break
            
            if scarlett_device is None:
                return False, "Scarlett 2i2 not found. Please connect it via USB and check that it's recognized by the system."
            
            # בדיקה שה-Scarlett תומך ב-192kHz
            try:
                sd.check_input_settings(device=scarlett_device['name'], 
                                      samplerate=192000, 
                                      channels=1)
                return True, f"Scarlett 2i2 ready: {scarlett_device['name']}"
            except Exception as e:
                # אם 192kHz לא עובד, ננסה 96kHz
                try:
                    sd.check_input_settings(device=scarlett_device['name'], 
                                          samplerate=96000, 
                                          channels=1)
                    return False, f"Scarlett found but only supports up to 96kHz. Please check USB connection and drivers."
                except:
                    return False, f"Scarlett found but has audio issues: {str(e)}"
                
        except Exception as e:
            return False, f"Error checking audio devices: {str(e)}"

    def start(self):
        # בדיקת כרטיסי קול לפני התחלה
        audio_ok, message = self.check_audio_devices()
        if not audio_ok:
            messagebox.showerror("Audio Device Error", 
                               f"Cannot start ultrasonic detection:\n\n{message}\n\n"
                               f"Please:\n"
                               f"1. Connect your Scarlett 2i2 via USB\n"
                               f"2. Check that it's recognized by the system\n"
                               f"3. Try again")
            return
        
        # הכל בסדר - מתחילים
        print(f"[INFO] {message}")
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Starting...")
        
        try:
            self.handle = lgpio.gpiochip_open(0)
            self.detector_thread = threading.Thread(target=self.detect_loop, daemon=True)
            self.detector_thread.start()
            self.status_label.config(text="Status: Running")
        except Exception as e:
            messagebox.showerror("GPIO Error", f"Failed to initialize GPIO:\n{str(e)}")
            self.running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Error")

    def stop(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")
        if self.handle is not None:
            lgpio.gpiochip_close(self.handle)
            self.handle = None

    def send_ttl(self, gpio, duration):
        lgpio.gpio_claim_output(self.handle, gpio)
        lgpio.gpio_write(self.handle, gpio, 1)
        time.sleep(duration)
        lgpio.gpio_write(self.handle, gpio, 0)

    def detect_loop(self):
        fs = 192000  # 192 kHz - מקסימום של Scarlett 2i2
        blocksize =16384#  4096  # בלוק גדול יותר לרזולוציה טובה

        def callback(indata, frames, time_info, status):
            if not self.running:
                raise sd.CallbackStop()
            ultrasonic_min_freq = max(0.0, float(self.min_freq_khz_var.get()) * 1000.0)
            # FFT
            audio = indata[:, 0]
            fft = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1/fs)
            # Find ultrasonic band
            mask = freqs >= ultrasonic_min_freq
            ultrasonic_fft = fft[mask]
            
            # בדיקה שיש נתונים לעבד
            if len(ultrasonic_fft) > 0:
                rms = np.sqrt(np.mean(np.abs(ultrasonic_fft)**2))
            else:
                rms = 0.0  # אין תדרים אולטרסוניים
                if not hasattr(self, '_last_debug_print_time'):
                    self._last_debug_print_time = 0
                now = time.time()
                if now - self._last_debug_print_time > 1:
                    print(f"[DEBUG] No ultrasonic frequencies found. Max freq: {freqs[-1]:.0f} Hz, Min needed: {ultrasonic_min_freq} Hz")
                    self._last_debug_print_time = now
            # עדכון GUI עם נתונים חיים
            self.master.after(0, lambda: self.rms_label.config(text=f"RMS: {rms:.4f}"))
            
            if rms > self.threshold_var.get():
                # עדכון GUI - זיהוי
                self.master.after(0, lambda: self.detection_label.config(text="Detection: MOUSE!", bg="red"))
                # Send TTL in a separate thread
                threading.Thread(
                    target=self.send_ttl,
                    args=(self.gpio_var.get(), self.ttl_duration_var.get()),
                    daemon=True
                ).start()
            else:
                # עדכון GUI - אין זיהוי
                self.master.after(0, lambda: self.detection_label.config(text="Detection: None", bg="green"))

        with sd.InputStream(channels=1, samplerate=fs, blocksize=blocksize, callback=callback):
            while self.running:
                time.sleep(0.1)

root = tk.Tk()
root.geometry("400x500")  # הגדלת החלון
root.title("Ultrasonic Mouse Detector")
app = UltrasonicDetectorApp(root)
root.mainloop()
