#!/usr/bin/env python3
"""
Unified GUI application for recording from dodotronic microphone and analyzing USV
Combines recording functionality with USV detection and visualization
"""

import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import threading
import sys
import os
from datetime import datetime

try:
    import sounddevice as sd
except OSError as e:
    if 'PortAudio' in str(e):
        print("=" * 80)
        print("ERROR: PortAudio library not found")
        print("=" * 80)
        print("\nPlease install PortAudio library:")
        print("  sudo apt-get install -y portaudio19-dev python3-pyaudio")
        print("\nThen reinstall sounddevice:")
        print("  pip install --upgrade --force-reinstall sounddevice")
        print("=" * 80)
        sys.exit(1)
    else:
        raise

import numpy as np
import scipy.io.wavfile as wav
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DodotronicRecorder:
    """Class for recording from dodotronic microphone"""
    def __init__(self, sample_rate=192000):
        self.sample_rate = sample_rate
        self.device_index = None
        self.device_name = None
        
    def find_dodotronic_device(self):
        """Search and identify dodotronic microphone"""
        devices = sd.query_devices()
        keywords = ['dodotronic', 'ultramic', '384k', '384', 'evo']
        
        for i, device in enumerate(devices):
            device_name_lower = device['name'].lower()
            is_input = device['max_input_channels'] > 0
            
            if is_input:
                for keyword in keywords:
                    if keyword in device_name_lower:
                        self.device_index = i
                        self.device_name = device['name']
                        if '384' in device_name_lower or 'ultramic' in device_name_lower:
                            self.sample_rate = 384000
                        return True
        return False
    
    def check_sample_rate_support(self):
        """Check support for sample rate and find best supported rate"""
        if self.device_index is None:
            return False
        
        try:
            sd.check_input_settings(
                device=self.device_index,
                samplerate=self.sample_rate,
                channels=1
            )
            return True
        except Exception as e:
            test_rates = [384000, 192000, 96000, 48000, 44100]
            for test_rate in test_rates:
                try:
                    sd.check_input_settings(
                        device=self.device_index,
                        samplerate=test_rate,
                        channels=1
                    )
                    self.sample_rate = test_rate
                    return True
                except:
                    continue
            return False
    
    def record(self, duration, progress_callback=None):
        """Record audio"""
        if self.device_index is None:
            if not self.find_dodotronic_device():
                raise RuntimeError("Dodotronic microphone not found")
        
        if not self.check_sample_rate_support():
            raise RuntimeError(f"Device does not support sample rate {self.sample_rate} Hz")
        
        num_samples = int(self.sample_rate * duration)
        
        stream_kwargs = {
            'samplerate': self.sample_rate,
            'channels': 1,
            'dtype': 'float32',
            'blocksize': 4096,
            'device': self.device_index
        }
        
        recording_data = sd.rec(frames=num_samples, **stream_kwargs)
        sd.wait()
        
        if len(recording_data.shape) > 1:
            recording_data = recording_data[:, 0]
        
        return recording_data


class USVRecorderAnalyzer:
    def __init__(self, master):
        self.master = master
        master.title("USV Recorder & Analyzer")
        master.geometry("800x1000")
        
        # Initialize recorder
        self.recorder = DodotronicRecorder()
        self.recorder.find_dodotronic_device()
        if self.recorder.device_index is not None:
            self.recorder.check_sample_rate_support()
        
        # GUI variables
        self.threshold_var = tk.DoubleVar(value=0.6)
        self.min_freq_khz_var = tk.DoubleVar(value=50.0)
        self.max_freq_khz_var = tk.DoubleVar(value=90.0)
        self.window_size_ms_var = tk.DoubleVar(value=100.0)
        self.hop_overlap_var = tk.DoubleVar(value=50.0)
        self.recording_duration_var = tk.DoubleVar(value=10.0)
        
        # Audio data
        self.audio_file = None
        self.audio_data = None
        self.sample_rate = None
        self.detection_results = None
        self.is_recording = False
        self.show_spec_bands_var = tk.BooleanVar(value=True)
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI layout - single page like USV.py"""
        # Title
        tk.Label(self.master, text="USV Recorder & Analyzer",
                font=("Arial", 16, "bold")).pack(pady=10)
        
        # --- Top section: Recording / Load ---
        top_frame = tk.Frame(self.master)
        top_frame.pack(pady=5, fill=tk.X)
        
        # Microphone info
        info_frame = tk.LabelFrame(top_frame, text="Microphone Info", padx=10, pady=5)
        info_frame.pack(pady=5, fill=tk.X)
        
        if self.recorder.device_index is not None:
            tk.Label(info_frame, text=f"Device: {self.recorder.device_name}",
                    font=("Arial", 10)).pack(anchor=tk.W)
            tk.Label(info_frame, text=f"Sample Rate: {self.recorder.sample_rate} Hz",
                    font=("Arial", 10)).pack(anchor=tk.W)
        else:
            tk.Label(info_frame, text="Dodotronic microphone not found",
                    fg="red", font=("Arial", 10)).pack(anchor=tk.W)
            tk.Button(info_frame, text="Refresh Device List",
                     command=self.refresh_device).pack(pady=5)
        
        # Recording + Load in one row
        record_load_frame = tk.Frame(top_frame)
        record_load_frame.pack(pady=5, fill=tk.X)
        
        tk.Label(record_load_frame, text="Duration (sec):").pack(side=tk.LEFT, padx=(0, 5))
        tk.Entry(record_load_frame, textvariable=self.recording_duration_var, width=10).pack(side=tk.LEFT, padx=5)
        self.record_button = tk.Button(record_load_frame, text="Record",
                                      command=self.start_recording, width=12)
        self.record_button.pack(side=tk.LEFT, padx=5)
        
        tk.Label(record_load_frame, text="  |  Load WAV:").pack(side=tk.LEFT, padx=(15, 5))
        tk.Button(record_load_frame, text="Browse WAV File", command=self.load_file, width=18).pack(side=tk.LEFT, padx=5)
        self.save_recording_btn = tk.Button(record_load_frame, text="Save Recording", command=self.save_recording, width=15, state=tk.DISABLED)
        self.save_recording_btn.pack(side=tk.LEFT, padx=5)
        
        # File / recording status (shows name and params after load or record)
        self.file_label = tk.Label(top_frame, text="No file selected", fg="red")
        self.file_label.pack(pady=5)
        
        # --- Analysis parameters (like USV.py) ---
        params_frame = tk.Frame(self.master)
        params_frame.pack(pady=10)
        
        tk.Label(params_frame, text="Analysis Parameters", font=("Arial", 12, "bold")).pack()
        
        tk.Label(params_frame, text="Ultrasonic Threshold (RMS):").pack(pady=5)
        tk.Entry(params_frame, textvariable=self.threshold_var).pack(pady=2)
        
        tk.Label(params_frame, text="Min Frequency (kHz):").pack(pady=5)
        tk.Entry(params_frame, textvariable=self.min_freq_khz_var).pack(pady=2)
        
        tk.Label(params_frame, text="Max Frequency (kHz):").pack(pady=5)
        tk.Entry(params_frame, textvariable=self.max_freq_khz_var).pack(pady=2)
        
        tk.Label(params_frame, text="Window Size (ms):").pack(pady=5)
        tk.Entry(params_frame, textvariable=self.window_size_ms_var).pack(pady=2)
        
        tk.Label(params_frame, text="Hop Overlap (%):").pack(pady=5)
        tk.Entry(params_frame, textvariable=self.hop_overlap_var).pack(pady=2)
        
        # Control buttons
        button_frame = tk.Frame(self.master)
        button_frame.pack(pady=10)
        
        self.analyze_button = tk.Button(button_frame, text="Analyze Audio",
                                        command=self.analyze_audio, width=15, state=tk.DISABLED)
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        
        self.save_results_button = tk.Button(button_frame, text="Save Results",
                                             command=self.save_results, width=15, state=tk.DISABLED)
        self.save_results_button.pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_label = tk.Label(self.master, text="Status: Ready", font=("Arial", 10, "bold"))
        self.status_label.pack(pady=5)
        
        # --- Results: 3 tabs (Text, Visualization, Spectrogram) ---
        results_frame = tk.Frame(self.master)
        results_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        tk.Label(results_frame, text="Analysis Results", font=("Arial", 12, "bold")).pack()
        
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(pady=5, fill=tk.BOTH, expand=True)
        
        # Text Results tab
        text_frame = tk.Frame(self.results_notebook)
        self.results_notebook.add(text_frame, text="Text Results")
        
        self.results_text = tk.Text(text_frame, height=8, width=60)
        scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Visualization tab
        viz_frame = tk.Frame(self.results_notebook)
        self.results_notebook.add(viz_frame, text="Visualization")
        
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Spectrogram tab
        spec_frame = tk.Frame(self.results_notebook)
        self.results_notebook.add(spec_frame, text="Spectrogram")
        
        spec_controls = tk.Frame(spec_frame)
        spec_controls.pack(fill=tk.X, pady=(0, 5))
        self.spec_show_bands_cb = tk.Checkbutton(
            spec_controls, text="Show detection bands",
            variable=self.show_spec_bands_var, command=self.create_spectrogram
        )
        self.spec_show_bands_cb.pack(anchor=tk.W)
        
        self.fig_spec, self.ax_spec = plt.subplots(figsize=(10, 6))
        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, spec_frame)
        self.canvas_spec.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.colorbar_spec = None
        
    def refresh_device(self):
        """Refresh microphone device list"""
        self.recorder = DodotronicRecorder()
        if self.recorder.find_dodotronic_device():
            self.recorder.check_sample_rate_support()
            self.status_label.config(text=f"Device found: {self.recorder.device_name}")
            messagebox.showinfo("Success", f"Microphone found:\n{self.recorder.device_name}\nSample Rate: {self.recorder.sample_rate} Hz")
        else:
            self.status_label.config(text="Device not found")
            messagebox.showwarning("Warning", "Dodotronic microphone not found")
    
    def start_recording(self):
        """Start recording"""
        if self.is_recording:
            return
        
        if self.recorder.device_index is None:
            messagebox.showerror("Error", "Microphone not found. Please check connection.")
            return
        
        duration = self.recording_duration_var.get()
        if duration <= 0:
            messagebox.showerror("Error", "Recording duration must be positive")
            return
        
        self.is_recording = True
        self.record_button.config(text="Recording...", state=tk.DISABLED)
        self.status_label.config(text=f"Recording for {duration} seconds...")
        
        # Start recording in separate thread
        self.recording_thread = threading.Thread(target=self.record_audio, args=(duration,), daemon=True)
        self.recording_thread.start()
    
    def record_audio(self, duration):
        """Record audio in background thread"""
        try:
            audio_data = self.recorder.record(duration)
            self.sample_rate = self.recorder.sample_rate
            
            # Update GUI
            self.master.after(0, lambda: self.recording_finished(audio_data))
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Recording Error", str(e)))
            self.master.after(0, self.recording_error)
    
    def recording_finished(self, audio_data):
        """Called when recording is finished"""
        self.is_recording = False
        self.audio_data = audio_data
        self.audio_file = None  # Not from file
        
        self.record_button.config(text="Record", state=tk.NORMAL)
        
        duration = len(audio_data) / self.sample_rate
        self.status_label.config(text=f"Recording completed: {duration:.2f}s, {self.sample_rate}Hz")
        
        # Enable analyze button
        self.analyze_button.config(state=tk.NORMAL)
        
        # Update file label with params
        self.file_label.config(text=f"Recording: {duration:.2f}s, {self.sample_rate} Hz", fg="green")
        self.save_recording_btn.config(state=tk.NORMAL)
        
        # Create spectrogram
        self.create_spectrogram()
        
        messagebox.showinfo("Success", f"Recording completed!\nDuration: {duration:.2f}s\nSample Rate: {self.sample_rate}Hz")
    
    def recording_error(self):
        """Called when recording error occurs"""
        self.is_recording = False
        self.record_button.config(text="Record", state=tk.NORMAL)
        self.status_label.config(text="Status: Recording failed")
    
    def load_file(self):
        """Load WAV file"""
        file_path = filedialog.askopenfilename(
            title="Select WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.audio_data, self.sample_rate = librosa.load(file_path, sr=None)
                self.audio_file = file_path
                
                filename = os.path.basename(file_path)
                duration = len(self.audio_data) / self.sample_rate
                self.file_label.config(text=f"Selected: {filename} — {duration:.2f}s, {self.sample_rate} Hz", fg="green")
                self.analyze_button.config(state=tk.NORMAL)
                self.save_recording_btn.config(state=tk.NORMAL)
                self.status_label.config(text=f"File loaded: {duration:.2f}s, {self.sample_rate}Hz")
                
                self.create_spectrogram()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
    
    def save_recording(self):
        """Save current recording"""
        if self.audio_data is None:
            messagebox.showwarning("No Data", "No recording to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Recording",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if not file_path.lower().endswith('.wav'):
                    file_path += '.wav'
                
                audio_normalized = np.clip(self.audio_data, -1.0, 1.0)
                audio_int16 = (audio_normalized * 32767).astype(np.int16)
                wav.write(file_path, self.sample_rate, audio_int16)
                
                messagebox.showinfo("Success", f"Recording saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save:\n{str(e)}")
    
    def analyze_audio(self):
        """Analyze audio for USV detection"""
        if self.audio_data is None:
            messagebox.showerror("Error", "No audio data loaded")
            return
        
        self.status_label.config(text="Status: Analyzing...")
        
        self.analysis_thread = threading.Thread(target=self.analyze_audio_data, daemon=True)
        self.analysis_thread.start()
    
    def analyze_audio_data(self):
        """Analyze audio data in background thread"""
        try:
            fs = self.sample_rate
            ultrasonic_min_freq = float(self.min_freq_khz_var.get()) * 1000.0
            ultrasonic_max_freq = float(self.max_freq_khz_var.get()) * 1000.0
            threshold = float(self.threshold_var.get())
            
            window_size_ms = float(self.window_size_ms_var.get())
            hop_overlap_percent = float(self.hop_overlap_var.get())
            
            window_size = int(fs * window_size_ms / 1000.0)
            hop_size = int(window_size * (1 - hop_overlap_percent / 100.0))
            
            detections = []
            total_windows = len(self.audio_data) // hop_size
            
            for i in range(0, len(self.audio_data) - window_size, hop_size):
                window = self.audio_data[i:i + window_size]
                fft = np.fft.rfft(window)
                freqs = np.fft.rfftfreq(len(window), 1/fs)
                
                mask = (freqs >= ultrasonic_min_freq) & (freqs <= ultrasonic_max_freq)
                ultrasonic_fft = fft[mask]
                
                if len(ultrasonic_fft) > 0:
                    rms = np.sqrt(np.mean(np.abs(ultrasonic_fft)**2))
                    
                    if rms > threshold:
                        start_time = i / fs
                        end_time = (i + window_size) / fs
                        detections.append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': end_time - start_time,
                            'rms': rms,
                            'max_freq': freqs[np.argmax(np.abs(fft))],
                            'ultrasonic_freqs': freqs[mask]
                        })
            
            self.detection_results = detections
            self.master.after(0, self.display_results)
            
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Analysis Error", f"Failed to analyze:\n{str(e)}"))
            self.master.after(0, lambda: self.status_label.config(text="Status: Error"))
    
    def display_results(self):
        """Display analysis results"""
        if not self.detection_results:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "No USV detections found.")
            self.status_label.config(text="Status: Analysis complete - No USV detected")
            self.create_visualization()
            self.create_spectrogram()
            return
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"USV Detection Results:\n")
        self.results_text.insert(tk.END, f"Total detections: {len(self.detection_results)}\n\n")
        
        for i, detection in enumerate(self.detection_results):
            self.results_text.insert(tk.END, f"Detection {i+1}:\n")
            self.results_text.insert(tk.END, f"  Time: {detection['start_time']:.3f}s - {detection['end_time']:.3f}s\n")
            self.results_text.insert(tk.END, f"  Duration: {detection['duration']:.3f}s\n")
            self.results_text.insert(tk.END, f"  RMS: {detection['rms']:.4f}\n")
            self.results_text.insert(tk.END, f"  Peak frequency: {detection['max_freq']:.1f} Hz\n\n")
        
        self.status_label.config(text=f"Status: Analysis complete - {len(self.detection_results)} USV detections")
        self.save_results_button.config(state=tk.NORMAL)
        
        self.create_visualization()
        self.create_spectrogram()
    
    def create_visualization(self):
        """Create audio visualization with USV detections"""
        if self.audio_data is None:
            return
        
        self.ax.clear()
        duration = len(self.audio_data) / self.sample_rate
        time_axis = np.linspace(0, duration, len(self.audio_data))
        
        downsample_factor = max(1, len(self.audio_data) // 10000)
        audio_downsampled = self.audio_data[::downsample_factor]
        time_downsampled = time_axis[::downsample_factor]
        
        self.ax.plot(time_downsampled, audio_downsampled, 'b-', alpha=0.7, linewidth=0.5, label='Audio Signal')
        
        if self.detection_results:
            for i, detection in enumerate(self.detection_results):
                start_time = detection['start_time']
                end_time = detection['end_time']
                self.ax.axvspan(start_time, end_time, alpha=0.3, color='red', 
                               label='USV Detection' if i == 0 else "")
                mid_time = (start_time + end_time) / 2
                self.ax.text(mid_time, max(audio_downsampled) * 0.8, f'{i+1}', 
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.7))
        
        self.ax.set_xlabel('Time (seconds)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('USV Detection Visualization')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.ax.set_xlim(0, duration)
        self.canvas.draw()
    
    def create_spectrogram(self):
        """Create spectrogram"""
        if self.audio_data is None or self.sample_rate is None:
            self.ax_spec.clear()
            if self.colorbar_spec is not None:
                try:
                    self.colorbar_spec.remove()
                except:
                    pass
                self.colorbar_spec = None
            self.ax_spec.set_title("No audio data")
            self.canvas_spec.draw()
            return
        
        try:
            for ax in self.fig_spec.axes:
                ax.remove()
            
            self.ax_spec = self.fig_spec.add_subplot(111)
            self.colorbar_spec = None
            
            n_fft = 2048
            hop_length = 512
            stft = librosa.stft(self.audio_data, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            spectrogram_db = librosa.amplitude_to_db(magnitude, ref=np.max)
            
            duration = len(self.audio_data) / self.sample_rate
            times = librosa.frames_to_time(np.arange(spectrogram_db.shape[1]), 
                                          sr=self.sample_rate, hop_length=hop_length)
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=n_fft)
            
            im = self.ax_spec.imshow(spectrogram_db, aspect='auto', origin='lower',
                                    extent=[times[0], times[-1], freqs[0], freqs[-1]],
                                    cmap='viridis', interpolation='bilinear')
            
            self.colorbar_spec = self.fig_spec.colorbar(im, ax=self.ax_spec, label='Magnitude (dB)')
            
            if self.detection_results and self.show_spec_bands_var.get():
                for detection in self.detection_results:
                    start_time = detection['start_time']
                    end_time = detection['end_time']
                    self.ax_spec.axvspan(start_time, end_time, alpha=0.3, color='red')
            
            self.ax_spec.set_xlabel('Time (seconds)')
            self.ax_spec.set_ylabel('Frequency (Hz)')
            self.ax_spec.set_title('Spectrogram - Raw Audio')
            self.ax_spec.set_xlim(0, duration)
            max_freq_display = min(freqs[-1], 100000)
            self.ax_spec.set_ylim(0, max_freq_display)
            self.canvas_spec.draw()
            
        except Exception as e:
            if not hasattr(self, 'ax_spec') or self.ax_spec not in self.fig_spec.axes:
                self.ax_spec = self.fig_spec.add_subplot(111)
            else:
                self.ax_spec.clear()
            self.ax_spec.set_title(f"Error creating spectrogram: {str(e)}")
            self.canvas_spec.draw()
    
    def save_results(self):
        """Save analysis results"""
        if not self.detection_results:
            messagebox.showwarning("No Results", "No results to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"USV Detection Results\n")
                    if self.audio_file:
                        f.write(f"File: {os.path.basename(self.audio_file)}\n")
                    else:
                        f.write(f"File: Recording\n")
                    f.write(f"Sample Rate: {self.sample_rate} Hz\n")
                    f.write(f"Threshold: {self.threshold_var.get()}\n")
                    f.write(f"Min Frequency: {self.min_freq_khz_var.get()} kHz\n")
                    f.write(f"Max Frequency: {self.max_freq_khz_var.get()} kHz\n")
                    f.write(f"Window Size: {self.window_size_ms_var.get()} ms\n")
                    f.write(f"Hop Overlap: {self.hop_overlap_var.get()}%\n")
                    f.write(f"Total detections: {len(self.detection_results)}\n\n")
                    
                    for i, detection in enumerate(self.detection_results):
                        f.write(f"Detection {i+1}:\n")
                        f.write(f"  Start Time: {detection['start_time']:.3f}s\n")
                        f.write(f"  End Time: {detection['end_time']:.3f}s\n")
                        f.write(f"  Duration: {detection['duration']:.3f}s\n")
                        f.write(f"  RMS: {detection['rms']:.4f}\n")
                        f.write(f"  Peak Frequency: {detection['max_freq']:.1f} Hz\n\n")
                
                messagebox.showinfo("Success", f"Results saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save results:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = USVRecorderAnalyzer(root)
    root.mainloop()
