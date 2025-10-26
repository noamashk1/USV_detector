import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import sounddevice as sd
import numpy as np
import time
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class UltrasonicDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title("Ultrasonic File Analyzer")

        # GUI variables
        self.threshold_var = tk.DoubleVar(value=0.2)
        self.gpio_var = tk.IntVar(value=17)
        self.ttl_duration_var = tk.DoubleVar(value=0.05)
        self.min_freq_khz_var = tk.DoubleVar(value=30.0)
        self.window_size_ms_var = tk.DoubleVar(value=100.0)  # גודל החלון במילישניות
        self.hop_overlap_var = tk.DoubleVar(value=50.0)  # חפיפה באחוזים
        self.running = False
        self.audio_file = None
        self.audio_data = None
        self.sample_rate = None
        self.detection_results = None

        # GUI layout
        tk.Label(master, text="Ultrasonic File Analyzer", font=("Arial", 16, "bold")).pack(pady=10)
        
        # File selection section
        file_frame = tk.Frame(master)
        file_frame.pack(pady=10)
        
        tk.Label(file_frame, text="Select WAV file:").pack()
        self.file_label = tk.Label(file_frame, text="No file selected", fg="red")
        self.file_label.pack(pady=5)
        
        tk.Button(file_frame, text="Browse WAV File", command=self.select_file, width=20).pack(pady=5)
        
        # Parameters section
        params_frame = tk.Frame(master)
        params_frame.pack(pady=10)
        
        tk.Label(params_frame, text="Analysis Parameters", font=("Arial", 12, "bold")).pack()
        
        tk.Label(params_frame, text="Ultrasonic Threshold (RMS):").pack(pady=5)
        tk.Entry(params_frame, textvariable=self.threshold_var).pack(pady=2)

        tk.Label(params_frame, text="Min Frequency (kHz):").pack(pady=5)
        tk.Entry(params_frame, textvariable=self.min_freq_khz_var).pack(pady=2)

        tk.Label(params_frame, text="GPIO Number:").pack(pady=5)
        tk.Entry(params_frame, textvariable=self.gpio_var).pack(pady=2)

        tk.Label(params_frame, text="TTL Duration (sec):").pack(pady=5)
        tk.Entry(params_frame, textvariable=self.ttl_duration_var).pack(pady=2)

        tk.Label(params_frame, text="Window Size (ms):").pack(pady=5)
        tk.Entry(params_frame, textvariable=self.window_size_ms_var).pack(pady=2)

        tk.Label(params_frame, text="Hop Overlap (%):").pack(pady=5)
        tk.Entry(params_frame, textvariable=self.hop_overlap_var).pack(pady=2)

        # Control buttons
        button_frame = tk.Frame(master)
        button_frame.pack(pady=10)
        
        self.analyze_button = tk.Button(button_frame, text="Analyze File", command=self.analyze_file, width=15, state=tk.DISABLED)
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        
        self.play_button = tk.Button(button_frame, text="Play Detection", command=self.play_detection, width=15, state=tk.DISABLED)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = tk.Button(button_frame, text="Save Results", command=self.save_results, width=15, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)

        # Status and results
        self.status_label = tk.Label(master, text="Status: Ready", font=("Arial", 10, "bold"))
        self.status_label.pack(pady=10)
        
        # Results display
        results_frame = tk.Frame(master)
        results_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        tk.Label(results_frame, text="Analysis Results", font=("Arial", 12, "bold")).pack()
        
        # Create notebook for tabs (results text and visualization)
        from tkinter import ttk
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(pady=5, fill=tk.BOTH, expand=True)
        
        # Results text tab
        self.results_frame = tk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Text Results")
        
        self.results_text = tk.Text(self.results_frame, height=8, width=60)
        self.results_text.pack(pady=5, fill=tk.BOTH, expand=True)
        
        # Visualization tab
        self.viz_frame = tk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Visualization")
        
        # Create matplotlib figure for visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = tk.Scale(master, from_=0, to=100, orient=tk.HORIZONTAL, 
                                   variable=self.progress_var, length=300, state=tk.DISABLED)
        self.progress_bar.pack(pady=5)

        self.detector_thread = None
        self.handle = None

    def select_file(self):
        """בחירת קובץ WAV לניתוח"""
        file_path = filedialog.askopenfilename(
            title="Select WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # טעינת הקובץ עם librosa
                self.audio_data, self.sample_rate = librosa.load(file_path, sr=None)
                self.audio_file = file_path
                
                # עדכון GUI
                filename = os.path.basename(file_path)
                self.file_label.config(text=f"Selected: {filename}", fg="green")
                self.analyze_button.config(state=tk.NORMAL)
                
                # הצגת מידע על הקובץ
                duration = len(self.audio_data) / self.sample_rate
                self.status_label.config(text=f"File loaded: {duration:.2f}s, {self.sample_rate}Hz")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
                self.file_label.config(text="Error loading file", fg="red")
                self.analyze_button.config(state=tk.DISABLED)

    def analyze_file(self):
        """ניתוח הקובץ לזיהוי USV"""
        if self.audio_data is None:
            messagebox.showerror("Error", "No file loaded")
            return
            
        self.status_label.config(text="Status: Analyzing...")
        self.progress_bar.config(state=tk.NORMAL)
        self.progress_var.set(0)
        
        # הפעלת הניתוח בתהליך נפרד
        self.detector_thread = threading.Thread(target=self.analyze_audio_data, daemon=True)
        self.detector_thread.start()

    def analyze_audio_data(self):
        """ניתוח הנתונים האודיו לזיהוי USV"""
        try:
            fs = self.sample_rate
            ultrasonic_min_freq = float(self.min_freq_khz_var.get()) * 1000.0
            threshold = float(self.threshold_var.get())
            
            # חלוקה לחלקים קטנים לניתוח - שימוש בפרמטרים מהממשק
            window_size_ms = float(self.window_size_ms_var.get())
            hop_overlap_percent = float(self.hop_overlap_var.get())
            
            window_size = int(fs * window_size_ms / 1000.0)  # המרה למילישניות
            hop_size = int(window_size * (1 - hop_overlap_percent / 100.0))  # חישוב חפיפה
            
            detections = []
            total_windows = len(self.audio_data) // hop_size
            
            for i in range(0, len(self.audio_data) - window_size, hop_size):
                # חישוב התקדמות
                progress = (i / hop_size) / total_windows * 100
                self.master.after(0, lambda p=progress: self.progress_var.set(p))
                
                # חיתוך חלון
                window = self.audio_data[i:i + window_size]
                
                # FFT
                fft = np.fft.rfft(window)
                freqs = np.fft.rfftfreq(len(window), 1/fs)
                
                # חיפוש תדרים אולטרסוניים
                mask = freqs >= ultrasonic_min_freq
                ultrasonic_fft = fft[mask]
                
                if len(ultrasonic_fft) > 0:
                    rms = np.sqrt(np.mean(np.abs(ultrasonic_fft)**2))
                    
                    if rms > threshold:
                        # זיהוי USV
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
            
            # עדכון GUI עם התוצאות
            self.master.after(0, self.display_results)
            
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Analysis Error", f"Failed to analyze file:\n{str(e)}"))
            self.master.after(0, lambda: self.status_label.config(text="Status: Error"))

    def display_results(self):
        """הצגת תוצאות הניתוח"""
        if not self.detection_results:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "No USV detections found in the file.")
            self.status_label.config(text="Status: Analysis complete - No USV detected")
            self.progress_bar.config(state=tk.DISABLED)
            self.create_visualization()  # עדיין ניצור גרף גם אם אין זיהויים
            return
        
        # הצגת התוצאות
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
        self.play_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.progress_bar.config(state=tk.DISABLED)
        
        # יצירת הוויזואליזציה
        self.create_visualization()

    def create_visualization(self):
        """יצירת ויזואליזציה של האודיו עם סימון זיהויי USV"""
        if self.audio_data is None:
            return
        
        # ניקוי הגרף הקודם
        self.ax.clear()
        
        # יצירת ציר זמן
        duration = len(self.audio_data) / self.sample_rate
        time_axis = np.linspace(0, duration, len(self.audio_data))
        
        # הצגת האודיו (downsampling לפרפורמנס טוב יותר)
        downsample_factor = max(1, len(self.audio_data) // 10000)  # מקסימום 10,000 נקודות
        audio_downsampled = self.audio_data[::downsample_factor]
        time_downsampled = time_axis[::downsample_factor]
        
        # ציור האודיו
        self.ax.plot(time_downsampled, audio_downsampled, 'b-', alpha=0.7, linewidth=0.5, label='Audio Signal')
        
        # סימון זיהויי USV
        if self.detection_results:
            for i, detection in enumerate(self.detection_results):
                start_time = detection['start_time']
                end_time = detection['end_time']
                
                # סימון אזור הזיהוי
                self.ax.axvspan(start_time, end_time, alpha=0.3, color='red', 
                               label='USV Detection' if i == 0 else "")
                
                # הוספת מספר זיהוי
                mid_time = (start_time + end_time) / 2
                self.ax.text(mid_time, max(audio_downsampled) * 0.8, f'{i+1}', 
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.7))
        
        # הגדרת הגרף
        self.ax.set_xlabel('Time (seconds)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('USV Detection Visualization')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # התאמת הגבולות
        self.ax.set_xlim(0, duration)
        
        # רענון הקנבס
        self.canvas.draw()

    def play_detection(self):
        """השמעת החלקים שזוהו כ-USV"""
        if not self.detection_results:
            messagebox.showwarning("No Detections", "No USV detections to play")
            return
        
        # יצירת אודיו משולב של כל הזיהויים
        combined_audio = []
        for detection in self.detection_results:
            start_idx = int(detection['start_time'] * self.sample_rate)
            end_idx = int(detection['end_time'] * self.sample_rate)
            segment = self.audio_data[start_idx:end_idx]
            combined_audio.extend(segment)
        
        if combined_audio:
            combined_audio = np.array(combined_audio)
            try:
                sd.play(combined_audio, self.sample_rate)
                self.status_label.config(text="Status: Playing USV detections...")
            except Exception as e:
                messagebox.showerror("Playback Error", f"Failed to play audio:\n{str(e)}")

    def save_results(self):
        """שמירת התוצאות לקובץ"""
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
                    f.write(f"File: {os.path.basename(self.audio_file)}\n")
                    f.write(f"Sample Rate: {self.sample_rate} Hz\n")
                    f.write(f"Threshold: {self.threshold_var.get()}\n")
                    f.write(f"Min Frequency: {self.min_freq_khz_var.get()} kHz\n")
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


root = tk.Tk()
root.geometry("600x900")  # הגדלת החלון לניתוח קבצים
root.title("Ultrasonic File Analyzer")
app = UltrasonicDetectorApp(root)
root.mainloop()
