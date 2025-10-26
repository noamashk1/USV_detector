import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf
from scipy import signal
import tkinter as tk
from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import os
import wave

class RecordingApp:
    def __init__(self, master):
        self.master = master
        master.title("USV Recording & Visualization")
        master.geometry("1000x700")
        
        # Recording parameters
        self.fs = 192000
        self.duration = 60
        self.recording_data = None
        self.is_recording = False
        self.is_playing_loop = False  # Flag to control loop playback
        self.playback_device = None  # Audio device for playback
        self.record_device = None  # Audio device for recording
        
        # Visualization parameters
        self.view_duration = 10.0  # seconds to display
        self.current_start_time = 0.0
        
        # Load loop audio file
        self.load_loop_audio()
        
        # Detect Scarlett device for playback
        self.detect_audio_device()
        
        # GUI setup
        self.setup_gui()
        
    def setup_gui(self):
        # Title
        tk.Label(self.master, text="USV Recording & Visualization", 
                font=("Arial", 16, "bold")).pack(pady=10)
        
        # Recording controls
        record_frame = tk.Frame(self.master)
        record_frame.pack(pady=10)
        
        tk.Label(record_frame, text="Recording Parameters", 
                font=("Arial", 12, "bold")).pack()
        
        params_frame = tk.Frame(record_frame)
        params_frame.pack(pady=5)
        
        tk.Label(params_frame, text="Duration (sec):").pack(side=tk.LEFT, padx=5)
        self.duration_var = tk.StringVar(value="60")
        tk.Entry(params_frame, textvariable=self.duration_var, width=10).pack(side=tk.LEFT, padx=5)
        
        tk.Label(params_frame, text="Sample Rate:").pack(side=tk.LEFT, padx=5)
        self.fs_var = tk.StringVar(value="192000")
        tk.Entry(params_frame, textvariable=self.fs_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Play loop checkbox
        self.play_loop_var = tk.BooleanVar(value=False)
        tk.Checkbutton(record_frame, text="Play USV loop during recording", 
                      variable=self.play_loop_var, font=("Arial", 10)).pack(pady=5)
        
        # Recording buttons
        button_frame = tk.Frame(record_frame)
        button_frame.pack(pady=10)
        
        self.record_button = tk.Button(button_frame, text="Start Recording", 
                                    command=self.start_recording, width=15)
        self.record_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = tk.Button(button_frame, text="Save Recording", 
                                   command=self.save_recording, width=15, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        self.load_button = tk.Button(button_frame, text="Load File", 
                                   command=self.load_file, width=15)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_label = tk.Label(self.master, text="Status: Ready", 
                                   font=("Arial", 10, "bold"))
        self.status_label.pack(pady=5)
        
        # Visualization controls
        viz_frame = tk.Frame(self.master)
        viz_frame.pack(pady=10, fill=tk.X)
        
        tk.Label(viz_frame, text="Visualization Controls", 
                font=("Arial", 12, "bold")).pack()
        
        controls_frame = tk.Frame(viz_frame)
        controls_frame.pack(pady=5)
        
        tk.Label(controls_frame, text="View Duration (sec):").pack(side=tk.LEFT, padx=5)
        self.view_duration_var = tk.StringVar(value="10")
        tk.Entry(controls_frame, textvariable=self.view_duration_var, width=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(controls_frame, text="Update View", 
                 command=self.update_view).pack(side=tk.LEFT, padx=5)
        
        # Time navigation
        nav_frame = tk.Frame(viz_frame)
        nav_frame.pack(pady=5)
        
        tk.Label(nav_frame, text="Time Navigation:").pack(side=tk.LEFT, padx=5)
        
        self.time_scale = tk.Scale(nav_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                 length=400, command=self.on_time_change)
        self.time_scale.pack(side=tk.LEFT, padx=5)
        
        # Matplotlib figure
        self.fig = Figure(figsize=(12, 4))
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initial empty plot
        self.update_plot()
    
    def detect_audio_device(self):
        """זיהוי כרטיס Scarlett להשמעה והקלטה"""
        try:
            devices = sd.query_devices()
            print("\nAvailable audio devices:")
            for i, device in enumerate(devices):
                print(f"  {i}: {device['name']} (inputs: {device['max_input_channels']}, outputs: {device['max_output_channels']})")
                if "Scarlett" in device["name"]:
                    if device["max_output_channels"] > 0:
                        self.playback_device = i
                        print(f"Found Scarlett output device: {device['name']} (index {i})")
                    if device["max_input_channels"] > 0:
                        self.record_device = i
                        print(f"Found Scarlett input device: {device['name']} (index {i})")
            print()
            if self.playback_device is None:
                print("Warning: Scarlett output device not found, using default device")
            if self.record_device is None:
                print("Warning: Scarlett input device not found, using default device")
        except Exception as e:
            print(f"Error detecting audio device: {e}")
    
    def load_loop_audio(self):
        """טעינת קובץ הלולאה"""
        self.loop_audio_file = "pup ICR USVs.wav"
        if not os.path.exists(self.loop_audio_file):
            self.loop_audio_data = None
            print(f"Warning: Loop audio file '{self.loop_audio_file}' not found")
            return
        
        try:
            # Load the loop audio file using soundfile (works better than wav.read)
            self.loop_audio_data, file_fs = sf.read(self.loop_audio_file)
            print(f"Loaded audio file: {file_fs} Hz")
            
            # Must use 192000 Hz to preserve ultrasonic frequencies (70 kHz USVs)
            # Recording at 192000 Hz captures up to 96 kHz
            target_fs = 192000  # Match recording sample rate
            if file_fs > target_fs:
                # Calculate resampling factor
                factor = target_fs / file_fs
                num_samples = int(len(self.loop_audio_data) * factor)
                print(f"Resampling from {file_fs} Hz to {target_fs} Hz ({factor:.2f}x slower) for playback")
                self.loop_audio_data = signal.resample(self.loop_audio_data, num_samples)
                self.loop_fs = target_fs
            elif file_fs != target_fs:
                # If file is already lower, just use it
                self.loop_fs = file_fs
                print(f"Using original sample rate: {file_fs} Hz")
            else:
                self.loop_fs = file_fs
                print(f"Using target sample rate: {file_fs} Hz")
            
            print(f"Loop audio ready: {self.loop_fs} Hz, {len(self.loop_audio_data)/self.loop_fs:.2f}s")
        except Exception as e:
            self.loop_audio_data = None
            print(f"Error loading loop audio: {e}")
    
    def create_human_tone(self, frequency=1000, duration=0.2, fs=44100):
        """יצירת צליל בן אדם (200ms, 1000 Hz)"""
        t = np.linspace(0, duration, int(fs * duration))
        tone = np.sin(2 * np.pi * frequency * t)
        return tone.astype(np.float32)
    
    def play_loop(self):
        """ניגון הלולאה ברקע"""
        if self.loop_audio_data is None:
            print("Loop audio data is None!")
            return
        
        print(f"Starting loop playback at {self.loop_fs} Hz...")
        
        # Create human-audible tone (200ms, 1000Hz at 44.1kHz)
        human_tone = self.create_human_tone(frequency=1000, duration=0.2, fs=44100)
        
        self.is_playing_loop = True
        
        try:
            # Use the same device as good_play_usv.py for Scarlett output
            loop_count = 0
            while self.is_playing_loop and self.is_recording:
                # Play human tone first so you can hear when new loop starts
                sd.play(human_tone, samplerate=44100)
                sd.wait()
                
                loop_count += 1
                print(f"Starting USV loop #{loop_count}")
                
                # Play ultrasonic audio
                if self.playback_device is not None:
                    print(f"Using Scarlett device {self.playback_device} for playback")
                    sd.play(self.loop_audio_data, samplerate=self.loop_fs, device=self.playback_device)
                else:
                    sd.play(self.loop_audio_data, samplerate=self.loop_fs)
                
                # Wait for playback to finish
                sd.wait()
                    
            print("Loop playback stopped")
        except Exception as e:
            print(f"Error playing loop: {e}")
            import traceback
            traceback.print_exc()
            self.is_playing_loop = False
    
    def stop_loop(self):
        """עצירת הלולאה"""
        self.is_playing_loop = False
        sd.stop()
        
    def start_recording(self):
        """התחלת הקלטה"""
        if self.is_recording:
            return
            
        try:
            self.duration = float(self.duration_var.get())
            self.fs = int(self.fs_var.get())
            
            self.is_recording = True
            self.record_button.config(text="Recording...", state=tk.DISABLED)
            status_msg = f"Recording for {self.duration} seconds..."
            if self.play_loop_var.get():
                status_msg += " (with USV loop)"
                print("Play loop enabled!")
                if self.loop_audio_data is None:
                    print("WARNING: Loop audio data is None but checkbox is checked!")
            self.status_label.config(text=status_msg)
            
            # Start playing loop if enabled
            if self.play_loop_var.get() and self.loop_audio_data is not None:
                print("Starting loop thread...")
                self.loop_thread = threading.Thread(target=self.play_loop, daemon=True)
                self.loop_thread.start()
                print("Loop thread started!")
            
            # Start recording in separate thread
            self.recording_thread = threading.Thread(target=self.record_audio, daemon=True)
            self.recording_thread.start()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for duration and sample rate")
            
    def record_audio(self):
        """הקלטת אודיו"""
        try:
            print("Recording..")
            # Use InputStream to allow simultaneous playback
            frames = int(self.fs * self.duration)
            # Don't specify device for input - let ALSA/PulseAudio handle duplex
            # This avoids "device busy" conflicts when playback is on the same device
            print(f"Recording from default input device (Scarlett will be used if available)")
            with sd.InputStream(samplerate=self.fs, channels=1, dtype='float32', blocksize=4096) as stream:
                self.recording_data = np.zeros((frames,), dtype='float32')
                total_read = 0
                while total_read < frames:
                    remaining = frames - total_read
                    to_read = min(remaining, 4096)
                    data, overflow = stream.read(to_read)
                    if overflow:
                        print(f"Warning: buffer overflow during recording")
                    self.recording_data[total_read:total_read+len(data)] = data[:, 0]
                    total_read += len(data)
                    # Print RMS every 5 seconds to verify ultrasonic detection
                    if total_read % (self.fs * 5) < 4096:
                        chunk_rms = np.sqrt(np.mean(self.recording_data[max(0, total_read-self.fs):total_read]**2))
                        # Check ultrasonic frequencies (70+ kHz)
                        fft = np.fft.rfft(self.recording_data[max(0, total_read-self.fs):total_read])
                        freqs = np.fft.rfftfreq(len(fft)*2-2, 1/self.fs)
                        usv_mask = freqs >= 70000
                        if len(fft[usv_mask]) > 0:
                            usv_rms = np.sqrt(np.mean(np.abs(fft[usv_mask])**2))
                            print(f"Recording progress: {total_read/frames*100:.0f}% - RMS: {chunk_rms:.4f}, USV RMS: {usv_rms:.2f}")
                        else:
                            print(f"Recording progress: {total_read/frames*100:.0f}% - RMS: {chunk_rms:.4f}, USV: not detected")
            print("Done")

            # Update GUI
            self.master.after(0, self.recording_finished)
            
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Recording Error", str(e)))
            self.master.after(0, self.recording_error)
            
    def recording_finished(self):
        """סיום הקלטה"""
        self.is_recording = False
        self.stop_loop()  # Stop the loop playback
        self.record_button.config(text="Start Recording", state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.status_label.config(text=f"Recording completed: {len(self.recording_data)/self.fs:.1f}s")
        
        # Update visualization
        self.update_time_scale()
        self.update_plot()
        
    def recording_error(self):
        """שגיאה בהקלטה"""
        self.is_recording = False
        self.stop_loop()  # Stop the loop playback
        self.record_button.config(text="Start Recording", state=tk.NORMAL)
        self.status_label.config(text="Status: Recording failed")
        
    def save_recording(self):
        """שמירת ההקלטה"""
        if self.recording_data is None:
            messagebox.showwarning("No Data", "No recording data to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Recording",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Convert to int16 for WAV file
                data_int16 = (self.recording_data * 32767).astype(np.int16)
                wav.write(file_path, self.fs, data_int16)
                messagebox.showinfo("Success", f"Recording saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save recording:\n{str(e)}")
                
    def load_file(self):
        """טעינת קובץ אודיו"""
        file_path = filedialog.askopenfilename(
            title="Load Audio File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.fs, self.recording_data = wav.read(file_path)
                # Convert to float32
                if self.recording_data.dtype == np.int16:
                    self.recording_data = self.recording_data.astype(np.float32) / 32767.0
                elif self.recording_data.dtype == np.int32:
                    self.recording_data = self.recording_data.astype(np.float32) / 2147483647.0
                
                self.save_button.config(state=tk.NORMAL)
                self.status_label.config(text=f"File loaded: {len(self.recording_data)/self.fs:.1f}s")
                
                # Update visualization
                self.update_time_scale()
                self.update_plot()
                
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load file:\n{str(e)}")
                
    def update_time_scale(self):
        """עדכון סרגל הזמן"""
        if self.recording_data is None:
            return
            
        total_duration = len(self.recording_data) / self.fs
        max_start_time = max(0, total_duration - self.view_duration)
        
        self.time_scale.config(from_=0, to=max_start_time, resolution=0.1)
        self.current_start_time = 0.0
        self.time_scale.set(0)
        
    def on_time_change(self, value):
        """שינוי זמן הצגה"""
        self.current_start_time = float(value)
        self.update_plot()
        
    def update_view(self):
        """עדכון משך הצגה"""
        try:
            self.view_duration = float(self.view_duration_var.get())
            self.update_time_scale()
            self.update_plot()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for view duration")
            
    def update_plot(self):
        """עדכון הגרף"""
        self.ax.clear()
        
        if self.recording_data is None:
            self.ax.set_title("No Data - Start Recording or Load File")
            self.ax.set_xlabel("Time (seconds)")
            self.ax.set_ylabel("Amplitude")
            self.canvas.draw()
            return
            
        # Calculate time range to display
        total_duration = len(self.recording_data) / self.fs
        start_sample = int(self.current_start_time * self.fs)
        end_sample = int(min((self.current_start_time + self.view_duration) * self.fs, 
                           len(self.recording_data)))
        
        # Extract data segment
        data_segment = self.recording_data[start_sample:end_sample]
        time_segment = np.linspace(self.current_start_time, 
                                 self.current_start_time + len(data_segment)/self.fs, 
                                 len(data_segment))
        
        # Downsample for better performance
        if len(data_segment) > 50000:
            downsample_factor = len(data_segment) // 50000
            data_segment = data_segment[::downsample_factor]
            time_segment = time_segment[::downsample_factor]
        
        # Plot
        self.ax.plot(time_segment, data_segment, 'b-', linewidth=0.5)
        self.ax.set_title(f"Audio Visualization - {self.current_start_time:.1f}s to {self.current_start_time + self.view_duration:.1f}s")
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.3)
        
        # Add time markers
        self.ax.axvline(self.current_start_time, color='red', linestyle='--', alpha=0.7, label='Start')
        self.ax.axvline(self.current_start_time + self.view_duration, color='red', linestyle='--', alpha=0.7, label='End')
        
        self.canvas.draw()

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = RecordingApp(root)
    root.mainloop()

