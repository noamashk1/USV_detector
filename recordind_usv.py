import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tkinter as tk
from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import os

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
        
        # Visualization parameters
        self.view_duration = 10.0  # seconds to display
        self.current_start_time = 0.0
        
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
        
    def start_recording(self):
        """התחלת הקלטה"""
        if self.is_recording:
            return
            
        try:
            self.duration = float(self.duration_var.get())
            self.fs = int(self.fs_var.get())
            
            self.is_recording = True
            self.record_button.config(text="Recording...", state=tk.DISABLED)
            self.status_label.config(text=f"Recording for {self.duration} seconds...")
            
            # Start recording in separate thread
            self.recording_thread = threading.Thread(target=self.record_audio, daemon=True)
            self.recording_thread.start()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for duration and sample rate")
            
    def record_audio(self):
        """הקלטת אודיו"""
        try:
            print("Recording..")
            self.recording_data = sd.rec(int(self.fs * self.duration), 
                                       samplerate=self.fs, channels=1, dtype='float32')
            sd.wait()
            print("Done")

            # Update GUI
            self.master.after(0, self.recording_finished)
            
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Recording Error", str(e)))
            self.master.after(0, self.recording_error)
            
    def recording_finished(self):
        """סיום הקלטה"""
        self.is_recording = False
        self.record_button.config(text="Start Recording", state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.status_label.config(text=f"Recording completed: {len(self.recording_data)/self.fs:.1f}s")
        
        # Update visualization
        self.update_time_scale()
        self.update_plot()
        
    def recording_error(self):
        """שגיאה בהקלטה"""
        self.is_recording = False
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

