#!/usr/bin/env python3
"""
GUI for USV detection in WAV files with a scrollable spectrogram viewport.

Designed for long recordings: audio stays on disk; only the visible time window
is read and rendered. Overview envelope and analysis run in background threads.
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle

OVERVIEW_TARGET_POINTS = 4000
SPEC_DEBOUNCE_MS = 200
ANALYSIS_CHUNK_SECONDS = 60.0
SPEC_HOP_LENGTH = 1024
SPEC_N_FFT = 2048


def read_wav_segment(
    path: str,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    """Read [start_sec, end_sec) from a WAV file without loading the whole file."""
    i0 = max(0, int(start_sec * sample_rate))
    i1 = max(i0 + 1, int(end_sec * sample_rate))
    data, _ = sf.read(path, start=i0, stop=i1, dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = np.mean(data, axis=1)
    return np.asarray(data, dtype=np.float32)


def build_overview_envelope(
    path: str,
    *,
    target_points: int = OVERVIEW_TARGET_POINTS,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Downsampled max-abs envelope for the navigation overview bar."""
    info = sf.info(path)
    sr = int(info.samplerate)
    duration = float(info.duration)
    total_frames = int(info.frames)
    if total_frames == 0:
        return np.array([0.0]), np.array([0.0]), sr, duration

    block_frames = max(1, total_frames // target_points)
    envelope: list[float] = []

    with sf.SoundFile(path, "r") as wav:
        blocks_total = max(1, (total_frames + block_frames - 1) // block_frames)
        for block_idx in range(blocks_total):
            block = wav.read(block_frames, dtype="float32", always_2d=False)
            if len(block) == 0:
                break
            if getattr(block, "ndim", 1) > 1:
                block = np.mean(block, axis=1)
            envelope.append(float(np.max(np.abs(block))))
            if progress_callback:
                progress_callback((block_idx + 1) / blocks_total)

    env = np.asarray(envelope, dtype=np.float32)
    times = np.linspace(0.0, duration, len(env), endpoint=True)
    return times, env, sr, duration


def detect_usv(
    audio: np.ndarray,
    fs: int,
    *,
    threshold: float,
    min_freq_hz: float,
    max_freq_hz: float,
    window_size_ms: float,
    hop_overlap_percent: float,
    progress_callback=None,
    progress_base: float = 0.0,
    progress_scale: float = 1.0,
) -> list[dict]:
    """Sliding-window RMS detection in an ultrasonic band (same logic as USV_recorder_analyzer)."""
    window_size = max(1, int(fs * window_size_ms / 1000.0))
    hop_size = max(1, int(window_size * (1.0 - hop_overlap_percent / 100.0)))

    detections: list[dict] = []
    n = len(audio)
    if n < window_size:
        return detections

    total_steps = max(1, (n - window_size) // hop_size)
    step = 0

    for i in range(0, n - window_size, hop_size):
        window = audio[i : i + window_size]
        fft = np.fft.rfft(window)
        freqs = np.fft.rfftfreq(len(window), 1.0 / fs)

        mask = (freqs >= min_freq_hz) & (freqs <= max_freq_hz)
        ultrasonic_fft = fft[mask]
        if len(ultrasonic_fft) == 0:
            step += 1
            if progress_callback:
                progress_callback(progress_base + progress_scale * (step / total_steps))
            continue

        rms = float(np.sqrt(np.mean(np.abs(ultrasonic_fft) ** 2)))
        if rms > threshold:
            start_time = i / fs
            end_time = (i + window_size) / fs
            detections.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "rms": rms,
                    "max_freq": float(freqs[np.argmax(np.abs(fft))]),
                }
            )

        step += 1
        if progress_callback:
            progress_callback(progress_base + progress_scale * (step / total_steps))

    return detections


def detect_usv_from_file(
    path: str,
    fs: int,
    total_frames: int,
    *,
    threshold: float,
    min_freq_hz: float,
    max_freq_hz: float,
    window_size_ms: float,
    hop_overlap_percent: float,
    progress_callback=None,
    chunk_seconds: float = ANALYSIS_CHUNK_SECONDS,
) -> list[dict]:
    """Run USV detection in time chunks read from disk."""
    window_size = max(1, int(fs * window_size_ms / 1000.0))
    chunk_frames = max(window_size * 2, int(chunk_seconds * fs))
    overlap_frames = window_size

    all_detections: list[dict] = []
    pos = 0
    num_chunks = max(1, (total_frames + chunk_frames - 1) // chunk_frames)

    while pos < total_frames:
        chunk_idx = pos // chunk_frames
        read_start = max(0, pos - overlap_frames) if pos > 0 else 0
        read_end = min(total_frames, pos + chunk_frames)
        audio = read_wav_segment(path, fs, read_start / fs, read_end / fs)

        def chunk_progress(p: float) -> None:
            if progress_callback:
                overall = (chunk_idx + p) / num_chunks
                progress_callback(min(overall, 1.0))

        chunk_dets = detect_usv(
            audio,
            fs,
            threshold=threshold,
            min_freq_hz=min_freq_hz,
            max_freq_hz=max_freq_hz,
            window_size_ms=window_size_ms,
            hop_overlap_percent=hop_overlap_percent,
            progress_callback=chunk_progress,
        )

        time_offset = read_start / fs
        chunk_boundary = pos / fs
        for det in chunk_dets:
            start_t = det["start_time"] + time_offset
            if pos > 0 and start_t < chunk_boundary - 1e-6:
                continue
            all_detections.append(
                {
                    "start_time": start_t,
                    "end_time": det["end_time"] + time_offset,
                    "duration": det["duration"],
                    "rms": det["rms"],
                    "max_freq": det["max_freq"],
                }
            )

        pos += chunk_frames

    return all_detections


def compute_spectrogram_db(
    segment: np.ndarray,
    sample_rate: int,
    start_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT magnitude (dB) for a visible audio segment."""
    stft = librosa.stft(segment, n_fft=SPEC_N_FFT, hop_length=SPEC_HOP_LENGTH)
    spec_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    rel_times = librosa.frames_to_time(
        np.arange(spec_db.shape[1]), sr=sample_rate, hop_length=SPEC_HOP_LENGTH
    ) + start_sec
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=SPEC_N_FFT)
    return spec_db, rel_times, freqs


class USVWavAnalyzerGUI:
    DEFAULT_VIEW_SECONDS = 5.0

    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        master.title("USV WAV Analyzer")
        master.minsize(1100, 720)
        master.geometry("1280x820")

        # Parameters
        self.threshold_var = tk.DoubleVar(value=0.6)
        self.min_freq_khz_var = tk.DoubleVar(value=50.0)
        self.max_freq_khz_var = tk.DoubleVar(value=90.0)
        self.window_size_ms_var = tk.DoubleVar(value=100.0)
        self.hop_overlap_var = tk.DoubleVar(value=50.0)
        self.view_seconds_var = tk.DoubleVar(value=self.DEFAULT_VIEW_SECONDS)
        self.view_start_var = tk.DoubleVar(value=0.0)

        self.show_usv_var = tk.BooleanVar(value=True)
        self.show_freq_band_var = tk.BooleanVar(value=True)

        self.audio_file: str | None = None
        self.sample_rate: int | None = None
        self.total_frames: int = 0
        self.duration: float = 0.0
        self.overview_times: np.ndarray | None = None
        self.overview_envelope: np.ndarray | None = None
        self.detection_results: list[dict] | None = None
        self.correction_mode = tk.StringVar(value="navigate")
        self._pending_add_start: float | None = None
        self._pending_add_line = None
        self._spec_click_cid: int | None = None
        self._review_widgets: list[tk.Widget] = []

        self._load_generation = 0
        self._spec_generation = 0
        self._spec_after_id: str | None = None
        self._spec_updating = False
        self._overview_rect: Rectangle | None = None
        self.colorbar = None

        self._build_ui()
        self.view_seconds_var.trace_add("write", lambda *_: self._on_view_length_changed())

    def _file_loaded(self) -> bool:
        return self.audio_file is not None and self.sample_rate is not None

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")

        outer = ttk.Panedwindow(self.master, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(outer, width=320)
        outer.add(left, weight=0)

        ttk.Label(left, text="USV WAV Analyzer", font=("Segoe UI", 14, "bold")).pack(
            anchor=tk.W, pady=(0, 8)
        )

        file_box = ttk.LabelFrame(left, text="WAV File", padding=8)
        file_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(file_box, text="Choose WAV file…", command=self.load_file).pack(fill=tk.X)
        self.file_label = ttk.Label(file_box, text="No file selected", foreground="#a00")
        self.file_label.pack(anchor=tk.W, pady=(6, 0))

        params = ttk.LabelFrame(left, text="USV Detection Parameters", padding=8)
        params.pack(fill=tk.X, pady=(0, 8))

        self._param_row(params, "RMS threshold:", self.threshold_var)
        self._param_row(params, "Min frequency (kHz):", self.min_freq_khz_var)
        self._param_row(params, "Max frequency (kHz):", self.max_freq_khz_var)
        self._param_row(params, "Window size (ms):", self.window_size_ms_var)
        self._param_row(params, "Window overlap (%):", self.hop_overlap_var)

        view_box = ttk.LabelFrame(left, text="Spectrogram Display", padding=8)
        view_box.pack(fill=tk.X, pady=(0, 8))
        self._param_row(view_box, "View window length (s):", self.view_seconds_var)

        display_box = ttk.LabelFrame(left, text="Plot Layers", padding=8)
        display_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Checkbutton(
            display_box,
            text="Show USV events",
            variable=self.show_usv_var,
            command=self.schedule_spectrogram,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            display_box,
            text="Show detection frequency band",
            variable=self.show_freq_band_var,
            command=self.schedule_spectrogram,
        ).pack(anchor=tk.W)

        btn_row = ttk.Frame(left)
        btn_row.pack(fill=tk.X, pady=(0, 8))
        self.analyze_btn = ttk.Button(
            btn_row, text="Analyze USV", command=self.run_analysis, state=tk.DISABLED
        )
        self.analyze_btn.pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(left, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, pady=(0, 6))

        self.status_label = ttk.Label(left, text="Ready", font=("Segoe UI", 10, "bold"))
        self.status_label.pack(anchor=tk.W)

        results_box = ttk.LabelFrame(left, text="Results (text)", padding=4)
        results_box.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.save_btn = ttk.Button(
            results_box,
            text="Save results to text file…",
            command=self.save_results,
            state=tk.DISABLED,
        )
        self.save_btn.pack(fill=tk.X, pady=(0, 4))
        self.results_text = tk.Text(results_box, height=12, width=36, wrap=tk.WORD, font=("Consolas", 9))
        scroll = ttk.Scrollbar(results_box, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scroll.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        right = ttk.Frame(outer)
        outer.add(right, weight=1)

        nav = ttk.LabelFrame(right, text="Recording Navigation", padding=6)
        nav.pack(fill=tk.X, pady=(0, 6))

        nav_top = ttk.Frame(nav)
        nav_top.pack(fill=tk.X)
        ttk.Label(nav_top, text="Position (seconds):").pack(side=tk.LEFT)
        self.time_pos_label = ttk.Label(nav_top, text="0.0 – 0.0 / 0.0")
        self.time_pos_label.pack(side=tk.LEFT, padx=(8, 0))

        self.nav_scale = ttk.Scale(
            nav,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.view_start_var,
            command=self._on_nav_changed,
        )
        self.nav_scale.pack(fill=tk.X, pady=(4, 6))

        nav_btns = ttk.Frame(nav)
        nav_btns.pack(fill=tk.X)
        ttk.Button(nav_btns, text="◀◀", width=4, command=lambda: self.nudge_view(-1.0)).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(nav_btns, text="◀", width=4, command=lambda: self.nudge_view(-0.25)).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(nav_btns, text="▶", width=4, command=lambda: self.nudge_view(0.25)).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(nav_btns, text="▶▶", width=4, command=lambda: self.nudge_view(1.0)).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(nav_btns, text="Jump to start", command=lambda: self.set_view_start(0.0)).pack(
            side=tk.LEFT, padx=(12, 2)
        )
        ttk.Button(nav_btns, text="Jump to end", command=self.jump_to_end).pack(side=tk.LEFT, padx=2)

        review_box = ttk.LabelFrame(right, text="Review detections (after Analyze)", padding=6)
        review_box.pack(fill=tk.X, pady=(0, 6))
        review_modes = ttk.Frame(review_box)
        review_modes.pack(fill=tk.X)
        for text, value in (
            ("Navigate", "navigate"),
            ("Remove false detection", "remove"),
            ("Add missed USV", "add"),
        ):
            rb = ttk.Radiobutton(
                review_modes,
                text=text,
                value=value,
                variable=self.correction_mode,
                command=self._on_correction_mode_changed,
                state=tk.DISABLED,
            )
            rb.pack(side=tk.LEFT, padx=(0, 10))
            self._review_widgets.append(rb)
        self.cancel_add_btn = ttk.Button(
            review_modes,
            text="Cancel start mark",
            command=self._cancel_pending_add,
            state=tk.DISABLED,
        )
        self.cancel_add_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.review_hint = ttk.Label(
            review_box,
            text="Run Analyze first. Remove: click a red mark. Add: start then end; Esc cancels start.",
            wraplength=720,
            foreground="#444",
        )
        self.review_hint.pack(anchor=tk.W, pady=(4, 0))

        self.master.bind("<Escape>", self._on_escape_key)

        self.fig_overview, self.ax_overview = plt.subplots(figsize=(8, 1.2))
        self.canvas_overview = FigureCanvasTkAgg(self.fig_overview, master=nav)
        self.canvas_overview.get_tk_widget().pack(fill=tk.X)
        self.canvas_overview.mpl_connect("button_press_event", self._on_overview_click)

        spec_frame = ttk.Frame(right)
        spec_frame.pack(fill=tk.BOTH, expand=True)

        self.fig_spec, self.ax_spec = plt.subplots(figsize=(9, 6))
        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=spec_frame)
        self.canvas_spec.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._draw_empty_plots()

    def _param_row(self, parent: ttk.Frame, label: str, var: tk.Variable) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=22).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var, width=10).pack(side=tk.RIGHT)

    def _draw_empty_plots(self) -> None:
        self.ax_overview.clear()
        self.ax_overview.set_title("Overview — click to jump")
        self.ax_overview.set_yticks([])
        self.canvas_overview.draw()

        self.ax_spec.clear()
        self.ax_spec.set_title("Spectrogram — load a WAV file")
        self.ax_spec.set_xlabel("Time (seconds)")
        self.ax_spec.set_ylabel("Frequency (Hz)")
        self.canvas_spec.draw()

    def _cancel_spec_debounce(self) -> None:
        if self._spec_after_id is not None:
            self.master.after_cancel(self._spec_after_id)
            self._spec_after_id = None

    # -------------------------------------------------------------- File I/O
    def load_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if not path:
            return

        self._cancel_spec_debounce()
        self._load_generation += 1
        self._spec_generation += 1
        load_gen = self._load_generation

        self.audio_file = None
        self.overview_times = None
        self.overview_envelope = None
        self.detection_results = None
        self._set_review_enabled(False)
        self.correction_mode.set("navigate")
        self._clear_pending_add_marker()
        self.analyze_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.progress["value"] = 0
        self.status_label.config(text="Loading file…")
        self.file_label.config(text=os.path.basename(path), foreground="#660")

        def worker() -> None:
            try:
                info = sf.info(path)
                sr = int(info.samplerate)
                duration = float(info.duration)
                frames = int(info.frames)

                def on_overview_progress(p: float) -> None:
                    if load_gen != self._load_generation:
                        return
                    self.master.after(
                        0,
                        lambda: self.progress.configure(value=5 + p * 45),
                    )

                times, env, _, _ = build_overview_envelope(
                    path, progress_callback=on_overview_progress
                )
                payload = {
                    "path": path,
                    "sr": sr,
                    "duration": duration,
                    "frames": frames,
                    "times": times,
                    "env": env,
                }
                self.master.after(
                    0, lambda: self._load_done(load_gen, payload, None)
                )
            except Exception as exc:
                self.master.after(
                    0, lambda: self._load_done(load_gen, None, exc)
                )

        threading.Thread(target=worker, daemon=True).start()

    def _load_done(
        self,
        load_gen: int,
        payload: dict | None,
        error: Exception | None,
    ) -> None:
        if load_gen != self._load_generation:
            return

        if error is not None:
            messagebox.showerror("Error", f"Could not load file:\n{error}")
            self.status_label.config(text="Load error")
            self.file_label.config(text="No file selected", foreground="#a00")
            self.progress["value"] = 0
            return

        assert payload is not None
        self.audio_file = payload["path"]
        self.sample_rate = payload["sr"]
        self.duration = payload["duration"]
        self.total_frames = payload["frames"]
        self.overview_times = payload["times"]
        self.overview_envelope = payload["env"]
        self.view_start_var.set(0.0)

        name = os.path.basename(self.audio_file)
        self.file_label.config(
            text=f"{name}\n{self.duration:.1f}s @ {self.sample_rate} Hz",
            foreground="#060",
        )
        self.analyze_btn.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.progress["value"] = 100
        self.status_label.config(text=f"Loaded: {self.duration:.1f}s")

        self._configure_navigation()
        self._draw_overview()
        self.schedule_spectrogram(immediate=True)

    def _on_view_length_changed(self) -> None:
        if not self._file_loaded():
            return
        self._configure_navigation()
        self.schedule_spectrogram()

    def _configure_navigation(self) -> None:
        view_len = self._view_length()
        max_start = max(0.0, self.duration - view_len)
        self.nav_scale.configure(from_=0.0, to=max_start if max_start > 0 else 0.0)
        if self.view_start_var.get() > max_start:
            self.view_start_var.set(max_start)
        self._update_time_label()

    def _view_length(self) -> float:
        try:
            v = float(self.view_seconds_var.get())
        except tk.TclError:
            v = self.DEFAULT_VIEW_SECONDS
        return max(1.0, v)

    def _view_start(self) -> float:
        try:
            start = float(self.view_start_var.get())
        except tk.TclError:
            start = 0.0
        max_start = max(0.0, self.duration - self._view_length())
        return float(np.clip(start, 0.0, max_start))

    def set_view_start(self, t: float) -> None:
        self._cancel_pending_add_on_navigate()
        self.view_start_var.set(t)
        self._configure_navigation()
        self.schedule_spectrogram()

    def _cancel_pending_add_on_navigate(self) -> None:
        if self._pending_add_start is not None:
            self._cancel_pending_add()

    def jump_to_end(self) -> None:
        self.set_view_start(max(0.0, self.duration - self._view_length()))

    def nudge_view(self, delta_fraction: float) -> None:
        self.set_view_start(self._view_start() + delta_fraction * self._view_length())

    def _on_nav_changed(self, _value: str) -> None:
        if self._spec_updating or not self._file_loaded():
            return
        self._update_time_label()
        self._update_overview_viewport()
        self.schedule_spectrogram()

    def _update_time_label(self) -> None:
        start = self._view_start()
        end = min(self.duration, start + self._view_length())
        self.time_pos_label.config(text=f"{start:.1f} – {end:.1f} / {self.duration:.1f}")

    # ---------------------------------------------------------- Visualization
    def _draw_overview(self) -> None:
        if (
            not self._file_loaded()
            or self.overview_times is None
            or self.overview_envelope is None
        ):
            return

        times = self.overview_times
        env = self.overview_envelope

        self.ax_overview.clear()
        self.ax_overview.fill_between(times, env, color="#4a90d9", alpha=0.55, linewidth=0)
        self.ax_overview.plot(times, env, color="#2c5282", linewidth=0.4)
        self.ax_overview.set_xlim(0, self.duration)
        self.ax_overview.set_ylim(0, max(float(env.max()) * 1.05, 1e-6))
        self.ax_overview.set_yticks([])
        self.ax_overview.set_xlabel("Time (seconds)")
        self.ax_overview.set_title(
            "Overview — cyan = view | red = auto USV | green = manual USV"
        )

        if self.detection_results and self.show_usv_var.get():
            for det in self.detection_results:
                self.ax_overview.axvspan(
                    det["start_time"],
                    det["end_time"],
                    color=self._detection_span_color(det),
                    alpha=0.25,
                    linewidth=0,
                )

        self._update_overview_viewport()
        self.fig_overview.tight_layout()
        self.canvas_overview.draw()

    def _update_overview_viewport(self) -> None:
        if not self._file_loaded():
            return
        start = self._view_start()
        end = min(self.duration, start + self._view_length())
        ymax = self.ax_overview.get_ylim()[1]

        if self._overview_rect is not None:
            try:
                self._overview_rect.remove()
            except Exception:
                pass
            self._overview_rect = None

        self._overview_rect = Rectangle(
            (start, 0),
            end - start,
            ymax,
            linewidth=1.5,
            edgecolor="#00838f",
            facecolor="#00bcd4",
            alpha=0.35,
        )
        self.ax_overview.add_patch(self._overview_rect)
        self.canvas_overview.draw_idle()

    def _on_overview_click(self, event) -> None:
        if event.inaxes != self.ax_overview or event.xdata is None:
            return
        if self.correction_mode.get() != "navigate":
            return
        half = self._view_length() / 2.0
        self.set_view_start(float(event.xdata) - half)

    def _set_review_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for widget in self._review_widgets:
            widget.config(state=state)
        if not enabled:
            self.cancel_add_btn.config(state=tk.DISABLED)
        else:
            self._update_cancel_add_btn()

    def _on_escape_key(self, _event=None) -> None:
        if self._pending_add_start is not None:
            self._cancel_pending_add()

    def _on_correction_mode_changed(self) -> None:
        self._cancel_pending_add(silent=True)
        mode = self.correction_mode.get()
        hints = {
            "navigate": "Navigate: use slider or overview (cyan box).",
            "remove": "Remove: click a red USV mark on the spectrogram.",
            "add": (
                "Add: click start time, then end time on the spectrogram. "
                "Cancel start: Esc or 'Cancel start mark'."
            ),
        }
        self.review_hint.config(text=hints.get(mode, ""))

    def _disconnect_spec_clicks(self) -> None:
        if self._spec_click_cid is not None:
            self.canvas_spec.mpl_disconnect(self._spec_click_cid)
            self._spec_click_cid = None

    def _connect_spec_clicks(self) -> None:
        self._disconnect_spec_clicks()
        if self._file_loaded():
            self._spec_click_cid = self.canvas_spec.mpl_connect(
                "button_press_event", self._on_spec_click
            )

    def _update_cancel_add_btn(self) -> None:
        if self._pending_add_start is not None and self.correction_mode.get() == "add":
            self.cancel_add_btn.config(state=tk.NORMAL)
        else:
            self.cancel_add_btn.config(state=tk.DISABLED)

    def _clear_pending_add_marker(self) -> None:
        self._pending_add_start = None
        if self._pending_add_line is not None:
            try:
                self._pending_add_line.remove()
            except Exception:
                pass
            self._pending_add_line = None
            self.canvas_spec.draw_idle()
        self._update_cancel_add_btn()

    def _cancel_pending_add(self, *, silent: bool = False) -> None:
        if self._pending_add_start is None:
            return
        self._clear_pending_add_marker()
        if not silent and self.correction_mode.get() == "add":
            self.status_label.config(text="Start mark cancelled — click start time again")

    def _detection_span_color(self, det: dict) -> str:
        return "#2ecc71" if det.get("manual") else "red"

    def _sort_detections(self) -> None:
        if self.detection_results:
            self.detection_results.sort(key=lambda d: d["start_time"])

    def _find_detection_index_at_time(self, time_sec: float) -> int | None:
        if not self.detection_results:
            return None
        for i, det in enumerate(self.detection_results):
            if det["start_time"] <= time_sec <= det["end_time"]:
                return i
        return None

    def _refresh_detections_ui(self) -> None:
        n = len(self.detection_results or [])
        self.save_btn.config(state=tk.NORMAL if n else tk.DISABLED)
        self._fill_results_text()
        self._draw_overview()
        self.schedule_spectrogram(immediate=True)

    def _remove_detection_at_time(self, time_sec: float) -> None:
        if not self.detection_results:
            return
        idx = self._find_detection_index_at_time(time_sec)
        if idx is None:
            self.status_label.config(text="No USV mark at this time in current list")
            return
        removed = self.detection_results.pop(idx)
        self._sort_detections()
        self.status_label.config(
            text=f"Removed #{idx + 1}: {removed['start_time']:.3f}s – {removed['end_time']:.3f}s"
        )
        self._refresh_detections_ui()

    def _default_manual_duration_sec(self) -> float:
        try:
            return max(0.05, float(self.window_size_ms_var.get()) / 1000.0)
        except tk.TclError:
            return 0.1

    def _add_manual_detection(self, start_sec: float, end_sec: float) -> None:
        if self.detection_results is None:
            self.detection_results = []
        t0, t1 = (start_sec, end_sec) if start_sec <= end_sec else (end_sec, start_sec)
        min_len = self._default_manual_duration_sec()
        if t1 - t0 < min_len:
            t1 = min(self.duration, t0 + min_len)
        t0 = max(0.0, t0)
        t1 = min(self.duration, t1)
        if t1 <= t0:
            self.status_label.config(text="Could not add event (invalid range)")
            return
        self.detection_results.append(
            {
                "start_time": t0,
                "end_time": t1,
                "duration": t1 - t0,
                "rms": float("nan"),
                "max_freq": float("nan"),
                "manual": True,
            }
        )
        self._sort_detections()
        self.status_label.config(text=f"Added manual USV: {t0:.3f}s – {t1:.3f}s")
        self._refresh_detections_ui()

    def _on_spec_click(self, event) -> None:
        if event.inaxes != self.ax_spec or event.xdata is None:
            return
        mode = self.correction_mode.get()
        if mode == "navigate":
            return
        if self.detection_results is None and mode == "remove":
            return

        time_sec = float(np.clip(event.xdata, 0.0, self.duration))

        if mode == "remove":
            self._remove_detection_at_time(time_sec)
            return

        if mode == "add":
            if self._pending_add_start is None:
                self._pending_add_start = time_sec
                self._pending_add_line = self.ax_spec.axvline(
                    time_sec, color="#2ecc71", ls="--", lw=1.5, alpha=0.95
                )
                self._update_cancel_add_btn()
                self.canvas_spec.draw_idle()
                self.status_label.config(
                    text=(
                        f"Start {time_sec:.3f}s — click end time "
                        "(Esc or Cancel start mark to undo)"
                    )
                )
            else:
                start_sec = self._pending_add_start
                self._clear_pending_add_marker()
                self._add_manual_detection(start_sec, time_sec)

    def schedule_spectrogram(self, *, immediate: bool = False) -> None:
        if not self._file_loaded():
            return
        self._cancel_spec_debounce()
        delay = 0 if immediate else SPEC_DEBOUNCE_MS
        self._spec_after_id = self.master.after(delay, self._start_spectrogram_worker)

    def _start_spectrogram_worker(self) -> None:
        self._spec_after_id = None
        if not self._file_loaded():
            return

        self._spec_generation += 1
        spec_gen = self._spec_generation
        path = self.audio_file
        sr = self.sample_rate
        start = self._view_start()
        view_len = self._view_length()
        end = min(self.duration, start + view_len)
        show_usv = self.show_usv_var.get()
        show_band = self.show_freq_band_var.get()
        detections = list(self.detection_results or [])
        try:
            fmin = float(self.min_freq_khz_var.get()) * 1000.0
            fmax = float(self.max_freq_khz_var.get()) * 1000.0
        except tk.TclError:
            fmin, fmax = 50_000.0, 90_000.0

        self.status_label.config(text="Rendering spectrogram…")

        def worker() -> None:
            try:
                segment = read_wav_segment(path, sr, start, end)
                if len(segment) < 256:
                    self.master.after(
                        0,
                        lambda: self._apply_spectrogram_error(
                            spec_gen, "Segment too short"
                        ),
                    )
                    return
                spec_db, rel_times, freqs = compute_spectrogram_db(segment, sr, start)
                payload = {
                    "spec_db": spec_db,
                    "rel_times": rel_times,
                    "freqs": freqs,
                    "start": start,
                    "end": end,
                    "show_usv": show_usv,
                    "show_band": show_band,
                    "fmin": fmin,
                    "fmax": fmax,
                    "detections": detections,
                }
                self.master.after(
                    0, lambda: self._apply_spectrogram(spec_gen, payload, None)
                )
            except Exception as exc:
                self.master.after(
                    0, lambda: self._apply_spectrogram(spec_gen, None, exc)
                )

        threading.Thread(target=worker, daemon=True).start()

    def _apply_spectrogram_error(self, spec_gen: int, message: str) -> None:
        if spec_gen != self._spec_generation:
            return
        self.ax_spec.clear()
        self.ax_spec.set_title(f"Spectrogram error: {message}")
        self.canvas_spec.draw_idle()
        if self._file_loaded():
            self.status_label.config(text=f"Loaded: {self.duration:.1f}s")

    def _apply_spectrogram(
        self,
        spec_gen: int,
        payload: dict | None,
        error: Exception | None,
    ) -> None:
        if spec_gen != self._spec_generation:
            return

        if error is not None:
            self._apply_spectrogram_error(spec_gen, str(error))
            return

        assert payload is not None
        self._spec_updating = True
        try:
            spec_db = payload["spec_db"]
            rel_times = payload["rel_times"]
            freqs = payload["freqs"]
            start = payload["start"]
            end = payload["end"]

            for ax in self.fig_spec.axes:
                ax.remove()
            self.ax_spec = self.fig_spec.add_subplot(111)
            self.colorbar = None

            im = self.ax_spec.imshow(
                spec_db,
                aspect="auto",
                origin="lower",
                extent=[rel_times[0], rel_times[-1], freqs[0], freqs[-1]],
                cmap="magma",
                interpolation="bilinear",
            )
            self.colorbar = self.fig_spec.colorbar(im, ax=self.ax_spec, label="Magnitude (dB)")

            if payload["show_band"]:
                self.ax_spec.axhline(
                    payload["fmin"], color="#00e5ff", ls="--", lw=1, alpha=0.9
                )
                self.ax_spec.axhline(
                    payload["fmax"], color="#00e5ff", ls="--", lw=1, alpha=0.9
                )

            if payload["show_usv"] and payload["detections"]:
                labeled_auto = False
                labeled_manual = False
                for det in payload["detections"]:
                    ds, de = det["start_time"], det["end_time"]
                    if de < start or ds > end:
                        continue
                    is_manual = det.get("manual", False)
                    label = None
                    if is_manual and not labeled_manual:
                        label = "Manual USV"
                        labeled_manual = True
                    elif not is_manual and not labeled_auto:
                        label = "Auto USV"
                        labeled_auto = True
                    self.ax_spec.axvspan(
                        max(ds, start),
                        min(de, end),
                        alpha=0.35,
                        color=self._detection_span_color(det),
                        label=label,
                    )

            self.ax_spec.set_xlim(start, end)
            max_freq_display = min(float(freqs[-1]), 100_000.0)
            self.ax_spec.set_ylim(0, max_freq_display)
            self.ax_spec.set_xlabel("Time (seconds)")
            self.ax_spec.set_ylabel("Frequency (Hz)")
            title = f"Spectrogram [{start:.1f}s – {end:.1f}s]"
            n_in_view = sum(
                1
                for d in payload["detections"]
                if d["end_time"] >= start and d["start_time"] <= end
            )
            if payload["detections"]:
                title += f" | {n_in_view} USV events in window"
            self.ax_spec.set_title(title)
            if payload["show_usv"] and payload["detections"]:
                handles, _ = self.ax_spec.get_legend_handles_labels()
                if handles:
                    self.ax_spec.legend(loc="upper right")

            self.fig_spec.tight_layout()
            self.canvas_spec.draw_idle()
            self._update_time_label()
            self._update_overview_viewport()
            self.status_label.config(text=f"Loaded: {self.duration:.1f}s")
            self._connect_spec_clicks()
        finally:
            self._spec_updating = False

    # --------------------------------------------------------------- Analysis
    def run_analysis(self) -> None:
        if not self._file_loaded():
            messagebox.showerror("Error", "Load a WAV file first")
            return

        try:
            threshold = float(self.threshold_var.get())
            min_hz = float(self.min_freq_khz_var.get()) * 1000.0
            max_hz = float(self.max_freq_khz_var.get()) * 1000.0
            if min_hz >= max_hz:
                raise ValueError("Minimum frequency must be lower than maximum frequency")
            window_ms = float(self.window_size_ms_var.get())
            hop_pct = float(self.hop_overlap_var.get())
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        path = self.audio_file
        sr = self.sample_rate
        total_frames = self.total_frames

        self.analyze_btn.config(state=tk.DISABLED)
        self.progress["value"] = 0
        self.status_label.config(text="Analyzing…")

        def worker() -> None:
            def on_progress(p: float) -> None:
                self.master.after(0, lambda: self.progress.configure(value=p * 100))

            try:
                manual_kept = [
                    dict(d)
                    for d in (self.detection_results or [])
                    if d.get("manual")
                ]
                results = detect_usv_from_file(
                    path,
                    sr,
                    total_frames,
                    threshold=threshold,
                    min_freq_hz=min_hz,
                    max_freq_hz=max_hz,
                    window_size_ms=window_ms,
                    hop_overlap_percent=hop_pct,
                    progress_callback=on_progress,
                )
                self.detection_results = results + manual_kept
                self._sort_detections()
                self.master.after(0, self._analysis_done, len(manual_kept))
            except Exception as exc:
                self.master.after(
                    0,
                    lambda: messagebox.showerror("Analysis error", str(exc)),
                )
                self.master.after(0, self._analysis_failed)

        threading.Thread(target=worker, daemon=True).start()

    def _analysis_done(self, manual_kept: int = 0) -> None:
        self.analyze_btn.config(state=tk.NORMAL)
        self.progress["value"] = 100
        n = len(self.detection_results or [])
        self._set_review_enabled(True)
        self.correction_mode.set("navigate")
        self._on_correction_mode_changed()
        status = f"Done — {n} detections. You can review marks below."
        if manual_kept:
            status += f" ({manual_kept} manual kept)"
        self.status_label.config(text=status)
        self.save_btn.config(state=tk.NORMAL if n else tk.DISABLED)
        self._fill_results_text()
        self._draw_overview()
        self.schedule_spectrogram(immediate=True)

    def _analysis_failed(self) -> None:
        self.analyze_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Analysis failed")

    def _format_results_report(self) -> str:
        """Text report of USV events (same content as the results pane)."""
        if not self.detection_results:
            return "No USV events found."

        lines = [f"Total detections: {len(self.detection_results)}\n"]
        for i, det in enumerate(self.detection_results, start=1):
            if det.get("manual"):
                lines.append(
                    f"#{i}: {det['start_time']:.3f}s – {det['end_time']:.3f}s (manual)\n"
                )
            else:
                lines.append(
                    f"#{i}: {det['start_time']:.3f}s – {det['end_time']:.3f}s\n"
                    f"    RMS={det['rms']:.4f}, peak={det['max_freq']:.0f} Hz\n"
                )
        return "\n".join(lines) + "\n"

    def _fill_results_text(self) -> None:
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, self._format_results_report())

    def save_results(self) -> None:
        if not self.detection_results:
            messagebox.showwarning("No results", "No results to save")
            return

        path = filedialog.asksaveasfilename(
            title="Save USV analysis results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            header_lines = ["USV analysis results", ""]
            if self.audio_file:
                header_lines.append(f"File: {os.path.basename(self.audio_file)}")
            if self.sample_rate is not None:
                header_lines.append(f"Sample rate: {self.sample_rate} Hz")
            header_lines.append(f"Duration: {self.duration:.3f} s")
            header_lines.append("")

            body = self._format_results_report()
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(header_lines))
                f.write(body)
            messagebox.showinfo("Saved", f"Results saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))


def main() -> None:
    root = tk.Tk()
    USVWavAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
