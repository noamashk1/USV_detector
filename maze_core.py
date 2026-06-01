import importlib.util
import json
import os
import shutil
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


def _prepare_opencv_qt_fonts() -> None:
    """OpenCV highgui on Linux uses Qt; recent wheels no longer bundle fonts."""
    if not sys.platform.startswith("linux"):
        return

    spec = importlib.util.find_spec("cv2")
    if spec is None or not spec.origin:
        return

    qt_fonts = Path(spec.origin).resolve().parent / "qt" / "fonts"
    if any(qt_fonts.glob("*.ttf")) or any(qt_fonts.glob("*.otf")):
        return

    system_dirs = (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/dejavu"),
        Path("/usr/share/fonts/truetype/liberation"),
        Path("/usr/share/fonts/TTF"),
    )
    for font_dir in system_dirs:
        if font_dir.is_dir():
            os.environ.setdefault("QT_QPA_FONTDIR", str(font_dir))
            break

    for font_dir in system_dirs:
        if not font_dir.is_dir():
            continue
        sources = [
            font_dir / name
            for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Regular.ttf")
            if (font_dir / name).is_file()
        ]
        if not sources:
            continue
        try:
            qt_fonts.mkdir(parents=True, exist_ok=True)
            for src in sources:
                dest = qt_fonts / src.name
                if not dest.exists():
                    shutil.copy2(src, dest)
        except OSError:
            pass
        break


_prepare_opencv_qt_fonts()

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

try:
    import lgpio
except ImportError:
    lgpio = None

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "maze_config.json"
ZONE_COLORS = [(0, 255, 0), (255, 128, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0), (128, 255, 128)]
ZONE_COLORS_RGB = ["#00ff00", "#ff8000", "#00c8ff", "#ff00ff", "#ffff00", "#80ff80"]


@dataclass
class RoiZone:
    name: str
    x: int
    y: int
    w: int
    h: int
    gpio: int
    background: np.ndarray | None = field(default=None, repr=False)
    consecutive_count: int = 0
    last_trigger_time: float = 0.0

    def rect(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def to_dict(self) -> dict:
        return {"name": self.name, "x": self.x, "y": self.y, "w": self.w, "h": self.h, "gpio": self.gpio}

    @classmethod
    def from_dict(cls, data: dict) -> "RoiZone":
        return cls(name=str(data["name"]), x=int(data["x"]), y=int(data["y"]), w=int(data["w"]), h=int(data["h"]), gpio=int(data["gpio"]))


@dataclass
class MazeConfig:
    frame_width: int = 640
    frame_height: int = 480
    motion_threshold: int = 500
    pulse_width_s: float = 0.05
    diff_threshold: int = 25
    blur_kernel: int = 21
    background_alpha: float = 0.02
    min_consecutive_frames: int = 2
    cooldown_s: float = 0.3
    morph_kernel_size: int = 3
    morph_open_iterations: int = 1
    morph_close_iterations: int = 1
    use_adaptive_threshold: bool = False
    adaptive_k_sigma: float = 2.5
    zones: list[RoiZone] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "motion_threshold": self.motion_threshold,
            "pulse_width_s": self.pulse_width_s,
            "diff_threshold": self.diff_threshold,
            "blur_kernel": self.blur_kernel,
            "background_alpha": self.background_alpha,
            "min_consecutive_frames": self.min_consecutive_frames,
            "cooldown_s": self.cooldown_s,
            "morph_kernel_size": self.morph_kernel_size,
            "morph_open_iterations": self.morph_open_iterations,
            "morph_close_iterations": self.morph_close_iterations,
            "use_adaptive_threshold": self.use_adaptive_threshold,
            "adaptive_k_sigma": self.adaptive_k_sigma,
            "zones": [z.to_dict() for z in self.zones],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MazeConfig":
        return cls(
            frame_width=int(data.get("frame_width", 640)),
            frame_height=int(data.get("frame_height", 480)),
            motion_threshold=int(data.get("motion_threshold", 500)),
            pulse_width_s=float(data.get("pulse_width_s", 0.05)),
            diff_threshold=int(data.get("diff_threshold", 25)),
            blur_kernel=int(data.get("blur_kernel", 21)),
            background_alpha=float(data.get("background_alpha", 0.02)),
            min_consecutive_frames=int(data.get("min_consecutive_frames", 2)),
            cooldown_s=float(data.get("cooldown_s", 0.3)),
            morph_kernel_size=int(data.get("morph_kernel_size", 3)),
            morph_open_iterations=int(data.get("morph_open_iterations", 1)),
            morph_close_iterations=int(data.get("morph_close_iterations", 1)),
            use_adaptive_threshold=bool(data.get("use_adaptive_threshold", False)),
            adaptive_k_sigma=float(data.get("adaptive_k_sigma", 2.5)),
            zones=[RoiZone.from_dict(z) for z in data.get("zones", [])],
        )


@dataclass
class RuntimeState:
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    latest_gray: np.ndarray | None = field(default=None, repr=False)
    running: bool = True


def load_config(path: Path) -> MazeConfig:
    with path.open(encoding="utf-8") as f:
        config = MazeConfig.from_dict(json.load(f))
    validate_config(config)
    return config


def save_config(config: MazeConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)


def validate_config(config: MazeConfig, frame_width: int | None = None, frame_height: int | None = None) -> None:
    if not config.zones:
        raise ValueError("Config must contain at least one zone")
    if config.blur_kernel % 2 == 0 or config.blur_kernel < 3:
        raise ValueError("blur_kernel must be odd and >= 3")
    if config.morph_kernel_size % 2 == 0 or config.morph_kernel_size < 3:
        raise ValueError("morph_kernel_size must be odd and >= 3")
    if not (0.0 < config.background_alpha <= 1.0):
        raise ValueError("background_alpha must be in (0, 1]")
    if config.min_consecutive_frames < 1:
        raise ValueError("min_consecutive_frames must be >= 1")
    if config.cooldown_s < 0:
        raise ValueError("cooldown_s must be >= 0")
    if config.morph_open_iterations < 0 or config.morph_close_iterations < 0:
        raise ValueError("morph iterations must be >= 0")
    if config.adaptive_k_sigma <= 0:
        raise ValueError("adaptive_k_sigma must be > 0")
    fw = frame_width if frame_width is not None else config.frame_width
    fh = frame_height if frame_height is not None else config.frame_height
    names: set[str] = set()
    gpios: set[int] = set()
    for zone in config.zones:
        if zone.w <= 0 or zone.h <= 0:
            raise ValueError(f"Zone '{zone.name}' has invalid size")
        if zone.name in names:
            raise ValueError(f"Duplicate zone name: {zone.name}")
        names.add(zone.name)
        if zone.gpio in gpios:
            raise ValueError(f"GPIO {zone.gpio} is assigned to more than one zone")
        gpios.add(zone.gpio)
        if zone.x < 0 or zone.y < 0 or zone.x + zone.w > fw or zone.y + zone.h > fh:
            raise ValueError(f"Zone '{zone.name}' is outside frame bounds ({fw}x{fh})")


class GpioManager:
    def __init__(self, pins: list[int], pulse_width_s: float):
        self._pulse_width_s = pulse_width_s
        self._pins = sorted(set(pins))
        self._lock = threading.Lock()
        self._handle = None
        self._available = False
        if not self._pins or lgpio is None:
            return
        self._handle = lgpio.gpiochip_open(0)
        for pin in self._pins:
            err = lgpio.gpio_claim_output(self._handle, pin)
            if err < 0:
                raise RuntimeError(f"lgpio gpio_claim_output failed for GPIO{pin}: {err}")
            lgpio.gpio_write(self._handle, pin, 0)
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def has_same_pins(self, pins: list[int]) -> bool:
        return sorted(set(pins)) == self._pins

    def pulse_async(self, pin: int) -> None:
        if not self._available:
            return
        threading.Thread(target=self._pulse, args=(pin,), daemon=True).start()

    def _pulse(self, pin: int) -> None:
        try:
            with self._lock:
                lgpio.gpio_write(self._handle, pin, 1)
            time.sleep(self._pulse_width_s)
        finally:
            with self._lock:
                lgpio.gpio_write(self._handle, pin, 0)

    def close(self) -> None:
        if self._handle is None:
            return
        for pin in self._pins:
            try:
                with self._lock:
                    lgpio.gpio_write(self._handle, pin, 0)
            except Exception:
                pass
        lgpio.gpiochip_close(self._handle)


def init_zone_runtime(zones: list[RoiZone], gray_frame: np.ndarray, blur_kernel: int) -> None:
    k = (blur_kernel, blur_kernel)
    for zone in zones:
        zone.background = cv2.GaussianBlur(gray_frame[zone.y : zone.y + zone.h, zone.x : zone.x + zone.w], k, 0).astype(np.float32)
        zone.consecutive_count = 0
        zone.last_trigger_time = 0.0


def process_zone(zone: RoiZone, gray: np.ndarray, config: MazeConfig) -> tuple[bool, np.ndarray, float]:
    roi = gray[zone.y : zone.y + zone.h, zone.x : zone.x + zone.w]
    roi_blurred = cv2.GaussianBlur(roi, (config.blur_kernel, config.blur_kernel), 0)
    # Zone size can change live while dragging; keep background shape aligned.
    if (
        zone.background is None
        or zone.background.shape[0] != roi_blurred.shape[0]
        or zone.background.shape[1] != roi_blurred.shape[1]
    ):
        zone.background = roi_blurred.astype(np.float32)
        zone.consecutive_count = 0
        zone.last_trigger_time = 0.0
    diff = cv2.absdiff(cv2.convertScaleAbs(zone.background), roi_blurred)
    threshold = max(float(config.diff_threshold), float(np.std(diff)) * config.adaptive_k_sigma) if config.use_adaptive_threshold else float(config.diff_threshold)
    mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((config.morph_kernel_size, config.morph_kernel_size), dtype=np.uint8)
    if config.morph_open_iterations > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=config.morph_open_iterations)
    if config.morph_close_iterations > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=config.morph_close_iterations)
    motion_amount = float(np.sum(mask) / 255.0)
    cv2.accumulateWeighted(roi_blurred.astype(np.float32), zone.background, config.background_alpha)
    return motion_amount > config.motion_threshold, mask, motion_amount


class RuntimeControlPanel:
    def __init__(self, config: MazeConfig, runtime: RuntimeState, frame_shape: tuple[int, int], on_gpio_change):
        self.config = config
        self.runtime = runtime
        self.frame_h, self.frame_w = frame_shape
        self.on_gpio_change = on_gpio_change

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        root = tk.Tk()
        root.title("Runtime Control")
        root.geometry("540x620")
        p = ttk.Frame(root, padding=10)
        p.pack(fill=tk.BOTH, expand=True)

        fields = {}
        for label, key in [
            ("motion_threshold", "motion_threshold"),
            ("diff_threshold", "diff_threshold"),
            ("background_alpha", "background_alpha"),
            ("min_consecutive_frames", "min_consecutive_frames"),
            ("cooldown_s", "cooldown_s"),
            ("morph_kernel_size", "morph_kernel_size"),
            ("morph_open_iterations", "morph_open_iterations"),
            ("morph_close_iterations", "morph_close_iterations"),
            ("adaptive_k_sigma", "adaptive_k_sigma"),
        ]:
            r = ttk.Frame(p)
            r.pack(fill=tk.X, pady=2)
            ttk.Label(r, text=label, width=22).pack(side=tk.LEFT)
            var = tk.StringVar()
            ttk.Entry(r, textvariable=var, width=14).pack(side=tk.LEFT)
            fields[key] = var

        use_adapt = tk.BooleanVar(value=False)
        ttk.Checkbutton(p, text="use_adaptive_threshold", variable=use_adapt).pack(anchor="w", pady=4)

        ttk.Label(p, text="Zones (name | x y w h gpio)").pack(anchor="w", pady=(10, 2))
        zones_list = tk.Listbox(p, height=10)
        zones_list.pack(fill=tk.BOTH, expand=True)

        edit = ttk.Frame(p)
        edit.pack(fill=tk.X, pady=6)
        z_name = tk.StringVar()
        z_x = tk.StringVar()
        z_y = tk.StringVar()
        z_w = tk.StringVar()
        z_h = tk.StringVar()
        z_gpio = tk.StringVar()
        for txt, var, width in [("name", z_name, 10), ("x", z_x, 4), ("y", z_y, 4), ("w", z_w, 4), ("h", z_h, 4), ("gpio", z_gpio, 5)]:
            ttk.Label(edit, text=txt).pack(side=tk.LEFT)
            ttk.Entry(edit, textvariable=var, width=width).pack(side=tk.LEFT, padx=2)

        status = tk.StringVar(value="")
        ttk.Label(p, textvariable=status).pack(anchor="w", pady=(2, 6))

        def refresh():
            with self.runtime.lock:
                fields["motion_threshold"].set(str(self.config.motion_threshold))
                fields["diff_threshold"].set(str(self.config.diff_threshold))
                fields["background_alpha"].set(str(self.config.background_alpha))
                fields["min_consecutive_frames"].set(str(self.config.min_consecutive_frames))
                fields["cooldown_s"].set(str(self.config.cooldown_s))
                fields["morph_kernel_size"].set(str(self.config.morph_kernel_size))
                fields["morph_open_iterations"].set(str(self.config.morph_open_iterations))
                fields["morph_close_iterations"].set(str(self.config.morph_close_iterations))
                fields["adaptive_k_sigma"].set(str(self.config.adaptive_k_sigma))
                use_adapt.set(bool(self.config.use_adaptive_threshold))
                zones = list(self.config.zones)
            zones_list.delete(0, tk.END)
            for z in zones:
                zones_list.insert(tk.END, f"{z.name} | {z.x} {z.y} {z.w} {z.h} {z.gpio}")

        def apply_params():
            try:
                with self.runtime.lock:
                    self.config.motion_threshold = int(fields["motion_threshold"].get())
                    self.config.diff_threshold = int(fields["diff_threshold"].get())
                    self.config.background_alpha = float(fields["background_alpha"].get())
                    self.config.min_consecutive_frames = int(fields["min_consecutive_frames"].get())
                    self.config.cooldown_s = float(fields["cooldown_s"].get())
                    self.config.morph_kernel_size = int(fields["morph_kernel_size"].get())
                    self.config.morph_open_iterations = int(fields["morph_open_iterations"].get())
                    self.config.morph_close_iterations = int(fields["morph_close_iterations"].get())
                    self.config.adaptive_k_sigma = float(fields["adaptive_k_sigma"].get())
                    self.config.use_adaptive_threshold = bool(use_adapt.get())
                    validate_config(self.config, frame_width=self.frame_w, frame_height=self.frame_h)
                status.set("Parameters applied")
            except Exception as exc:
                status.set(f"Error: {exc}")

        def load_selected():
            idx = zones_list.curselection()
            if not idx:
                return
            with self.runtime.lock:
                z = self.config.zones[idx[0]]
            z_name.set(z.name); z_x.set(str(z.x)); z_y.set(str(z.y)); z_w.set(str(z.w)); z_h.set(str(z.h)); z_gpio.set(str(z.gpio))

        def _new_zone_from_inputs() -> RoiZone:
            return RoiZone(name=z_name.get().strip(), x=int(z_x.get()), y=int(z_y.get()), w=int(z_w.get()), h=int(z_h.get()), gpio=int(z_gpio.get()))

        def _init_zone_bg(z: RoiZone):
            if self.runtime.latest_gray is None:
                return
            roi = self.runtime.latest_gray[z.y : z.y + z.h, z.x : z.x + z.w]
            if roi.size == 0:
                return
            z.background = cv2.GaussianBlur(roi, (self.config.blur_kernel, self.config.blur_kernel), 0).astype(np.float32)
            z.consecutive_count = 0
            z.last_trigger_time = 0.0

        def add_zone():
            try:
                nz = _new_zone_from_inputs()
                with self.runtime.lock:
                    self.config.zones.append(nz)
                    validate_config(self.config, frame_width=self.frame_w, frame_height=self.frame_h)
                    _init_zone_bg(nz)
                    self.on_gpio_change()
                refresh()
                status.set("Zone added")
            except Exception as exc:
                status.set(f"Error: {exc}")

        def update_zone():
            idx = zones_list.curselection()
            if not idx:
                status.set("Select zone first")
                return
            try:
                uz = _new_zone_from_inputs()
                with self.runtime.lock:
                    self.config.zones[idx[0]] = uz
                    validate_config(self.config, frame_width=self.frame_w, frame_height=self.frame_h)
                    _init_zone_bg(uz)
                    self.on_gpio_change()
                refresh()
                status.set("Zone updated")
            except Exception as exc:
                status.set(f"Error: {exc}")

        def delete_zone():
            idx = zones_list.curselection()
            if not idx:
                return
            with self.runtime.lock:
                del self.config.zones[idx[0]]
                if not self.config.zones:
                    status.set("Need at least one zone")
                    return
                self.on_gpio_change()
            refresh()
            status.set("Zone deleted")

        btns = ttk.Frame(p)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Apply params", command=apply_params).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Load selected", command=load_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Add zone", command=add_zone).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Update zone", command=update_zone).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Delete zone", command=delete_zone).pack(side=tk.LEFT, padx=2)

        refresh()
        root.protocol("WM_DELETE_WINDOW", root.withdraw)
        root.mainloop()


class MazeSetupGUI:
    def __init__(self, first_frame: np.ndarray):
        if ImageTk is None:
            raise RuntimeError("Pillow is required for image display in GUI. Install: pip install Pillow")
        h, w = first_frame.shape[:2]
        self._frame_bgr = first_frame.copy()
        self.config = MazeConfig(frame_width=w, frame_height=h)
        self.config_path = DEFAULT_CONFIG_PATH
        self.result: tuple[MazeConfig, Path] | None = None
        self._save_on_start = False
        self._scale = 1.0
        self._photo = None
        self._drag_start = None
        self._pending_roi = None
        self._rect_item = None
        self.root = tk.Tk()
        self.root.title("Maze — Setup")
        self._build_ui()
        self._show_mode_screen()

    def _build_ui(self):
        self.container = ttk.Frame(self.root, padding=12); self.container.pack(fill=tk.BOTH, expand=True)
        self.mode_frame = ttk.Frame(self.container); self.editor_frame = ttk.Frame(self.container)
        ttk.Label(self.mode_frame, text="Maze Motion Detector", font=("", 14, "bold")).pack(anchor="w", pady=(0, 16))
        ttk.Button(self.mode_frame, text="Load settings from JSON", command=self._on_load_json).pack(anchor="w", pady=4)
        ttk.Button(self.mode_frame, text="Interactive setup", command=self._on_interactive).pack(anchor="w", pady=4)
        ttk.Button(self.mode_frame, text="Exit", command=self._cancel).pack(anchor="w", pady=8)
        top = ttk.Frame(self.editor_frame); top.pack(fill=tk.BOTH, expand=True)
        canvas_frame = ttk.LabelFrame(top, text="Camera — drag rectangle to select ROI", padding=4); canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        self.canvas = tk.Canvas(canvas_frame, bg="#222", highlightthickness=0); self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_press); self.canvas.bind("<B1-Motion>", self._on_drag); self.canvas.bind("<ButtonRelease-1>", self._on_release)
        side = ttk.Frame(top, width=320); side.pack(side=tk.RIGHT, fill=tk.Y); side.pack_propagate(False)
        self.vars = {}
        for label, key, default in [
            ("Motion threshold:", "motion_threshold", "500"), ("Pulse width (s):", "pulse_width_s", "0.05"), ("Diff threshold:", "diff_threshold", "25"),
            ("Blur kernel:", "blur_kernel", "21"), ("Background alpha:", "background_alpha", "0.02"), ("Min consecutive frames:", "min_consecutive_frames", "2"),
            ("Cooldown (s):", "cooldown_s", "0.3"), ("Morph kernel size:", "morph_kernel_size", "3"), ("Morph open iterations:", "morph_open_iterations", "1"),
            ("Morph close iterations:", "morph_close_iterations", "1"), ("Adaptive k sigma:", "adaptive_k_sigma", "2.5"),
        ]:
            r = ttk.Frame(side); r.pack(fill=tk.X, pady=1)
            ttk.Label(r, text=label).pack(side=tk.LEFT)
            self.vars[key] = tk.StringVar(value=default)
            ttk.Entry(r, textvariable=self.vars[key], width=10).pack(side=tk.LEFT, padx=4)
        self.use_adaptive = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="Use adaptive threshold", variable=self.use_adaptive).pack(anchor="w", pady=(2, 8))
        zone_box = ttk.LabelFrame(side, text="New zone", padding=6); zone_box.pack(fill=tk.X, pady=8)
        self.name_var = tk.StringVar(value="zone_1"); self.gpio_var = tk.StringVar(value="17")
        ttk.Label(zone_box, text="Zone name").pack(anchor="w"); ttk.Entry(zone_box, textvariable=self.name_var).pack(fill=tk.X)
        ttk.Label(zone_box, text="GPIO").pack(anchor="w"); ttk.Entry(zone_box, textvariable=self.gpio_var).pack(fill=tk.X)
        ttk.Button(zone_box, text="Add zone from selection", command=self._add_zone).pack(anchor="w", pady=6)
        self.zone_list = tk.Listbox(side, height=8); self.zone_list.pack(fill=tk.BOTH, expand=True)
        ttk.Button(side, text="Remove selected zone", command=self._remove_zone).pack(anchor="w", pady=4)
        self.status_var = tk.StringVar(value=""); ttk.Label(self.editor_frame, textvariable=self.status_var).pack(anchor="w")
        b = ttk.Frame(self.editor_frame); b.pack(fill=tk.X, pady=6)
        ttk.Button(b, text="Start monitoring", command=self._on_start).pack(side=tk.LEFT, padx=4)
        ttk.Button(b, text="Save JSON...", command=self._save_as).pack(side=tk.LEFT, padx=4)
        ttk.Button(b, text="Back", command=self._show_mode_screen).pack(side=tk.LEFT, padx=4)
        ttk.Button(b, text="Exit", command=self._cancel).pack(side=tk.RIGHT, padx=4)

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._cancel); self.root.mainloop(); return self.result
    def _show_mode_screen(self): self.editor_frame.pack_forget(); self.mode_frame.pack(fill=tk.BOTH, expand=True); self.root.geometry("500x220")
    def _show_editor_screen(self): self.mode_frame.pack_forget(); self.editor_frame.pack(fill=tk.BOTH, expand=True); self.root.geometry("1180x760"); self._render_canvas(); self._refresh_zones()

    def _on_load_json(self):
        path = filedialog.askopenfilename(title="Select settings file", initialdir=str(SCRIPT_DIR), filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path: return
        self.config = load_config(Path(path)); self.config_path = Path(path); self._save_on_start = False; self._sync_from_config(); self._show_editor_screen()
    def _on_interactive(self):
        h, w = self._frame_bgr.shape[:2]; self.config = MazeConfig(frame_width=w, frame_height=h); self.config_path = DEFAULT_CONFIG_PATH; self._save_on_start = True
        self._sync_from_config(); self._show_editor_screen()
    def _sync_from_config(self):
        for k in self.vars: self.vars[k].set(str(getattr(self.config, k)))
        self.use_adaptive.set(bool(self.config.use_adaptive_threshold))
    def _apply_to_config(self):
        self.config.motion_threshold = int(self.vars["motion_threshold"].get()); self.config.pulse_width_s = float(self.vars["pulse_width_s"].get())
        self.config.diff_threshold = int(self.vars["diff_threshold"].get()); self.config.blur_kernel = int(self.vars["blur_kernel"].get())
        self.config.background_alpha = float(self.vars["background_alpha"].get()); self.config.min_consecutive_frames = int(self.vars["min_consecutive_frames"].get())
        self.config.cooldown_s = float(self.vars["cooldown_s"].get()); self.config.morph_kernel_size = int(self.vars["morph_kernel_size"].get())
        self.config.morph_open_iterations = int(self.vars["morph_open_iterations"].get()); self.config.morph_close_iterations = int(self.vars["morph_close_iterations"].get())
        self.config.adaptive_k_sigma = float(self.vars["adaptive_k_sigma"].get()); self.config.use_adaptive_threshold = bool(self.use_adaptive.get())

    def _render_canvas(self):
        h, w = self._frame_bgr.shape[:2]; self._scale = min(760 / w, 560 / h, 1.0); dw, dh = max(1, int(w * self._scale)), max(1, int(h * self._scale))
        rgb = cv2.cvtColor(cv2.resize(self._frame_bgr, (dw, dh)), cv2.COLOR_BGR2RGB); self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.config(width=dw, height=dh); self.canvas.delete("all"); self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
        for i, z in enumerate(self.config.zones):
            color = ZONE_COLORS_RGB[i % len(ZONE_COLORS_RGB)]
            x, y, ww, hh = int(z.x * self._scale), int(z.y * self._scale), int(z.w * self._scale), int(z.h * self._scale)
            self.canvas.create_rectangle(x, y, x + ww, y + hh, outline=color, width=2)
            self.canvas.create_text(x + 4, y + 14, anchor=tk.NW, text=f"{z.name} GPIO{z.gpio}", fill=color)
    def _refresh_zones(self):
        self.zone_list.delete(0, tk.END)
        for z in self.config.zones: self.zone_list.insert(tk.END, f"{z.name} | ({z.x},{z.y}) {z.w}x{z.h} | GPIO{z.gpio}")
    def _canvas_to_frame(self, cx, cy): return int(cx / self._scale), int(cy / self._scale)
    def _on_press(self, e): self._drag_start = (e.x, e.y)
    def _on_drag(self, e):
        if self._drag_start is None: return
        x0, y0 = self._drag_start
        if self._rect_item is not None: self.canvas.delete(self._rect_item)
        self._rect_item = self.canvas.create_rectangle(x0, y0, e.x, e.y, outline="#ffff00", width=2)
    def _on_release(self, e):
        if self._drag_start is None: return
        x0, y0 = self._drag_start; self._drag_start = None
        fx0, fy0 = self._canvas_to_frame(min(x0, e.x), min(y0, e.y)); fx1, fy1 = self._canvas_to_frame(max(x0, e.x), max(y0, e.y))
        w, h = fx1 - fx0, fy1 - fy0
        self._pending_roi = (fx0, fy0, w, h) if w > 5 and h > 5 else None
    def _add_zone(self):
        if self._pending_roi is None: return
        x, y, w, h = self._pending_roi
        self.config.zones.append(RoiZone(name=self.name_var.get().strip() or f"zone_{len(self.config.zones)+1}", x=x, y=y, w=w, h=h, gpio=int(self.gpio_var.get())))
        self._render_canvas(); self._refresh_zones()
    def _remove_zone(self):
        sel = self.zone_list.curselection()
        if not sel: return
        del self.config.zones[sel[0]]
        self._render_canvas(); self._refresh_zones()
    def _save_as(self):
        self._apply_to_config(); validate_config(self.config, frame_width=self.config.frame_width, frame_height=self.config.frame_height)
        p = filedialog.asksaveasfilename(title="Save settings", initialdir=str(SCRIPT_DIR), initialfile="maze_config.json", defaultextension=".json", filetypes=[("JSON", "*.json")])
        if p: save_config(self.config, Path(p)); self.config_path = Path(p)
    def _on_start(self):
        self._apply_to_config(); validate_config(self.config, frame_width=self.config.frame_width, frame_height=self.config.frame_height)
        if self._save_on_start: save_config(self.config, self.config_path)
        self.result = (self.config, self.config_path); self.root.destroy()
    def _cancel(self): self.result = None; self.root.destroy()


def draw_zones(frame: np.ndarray, zones: list[RoiZone], active: dict[str, bool], debug: dict[str, str]) -> None:
    for i, zone in enumerate(zones):
        color = ZONE_COLORS[i % len(ZONE_COLORS)]
        cv2.rectangle(frame, (zone.x, zone.y), (zone.x + zone.w, zone.y + zone.h), color, 2)
        cv2.putText(frame, f"{zone.name} GPIO{zone.gpio}", (zone.x, max(zone.y - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if active.get(zone.name):
            cv2.putText(frame, "MOTION", (zone.x + 4, zone.y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, debug.get(zone.name, ""), (zone.x + 4, min(zone.y + zone.h - 8, frame.shape[0] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def open_camera(config: MazeConfig) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
    return cap


def run_monitoring(cap: cv2.VideoCapture, config: MazeConfig, first_frame: np.ndarray) -> None:
    runtime = RuntimeState()
    gray0 = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    init_zone_runtime(config.zones, gray0, config.blur_kernel)
    runtime.latest_gray = gray0
    gpio_lock = threading.Lock()
    gpio = GpioManager([z.gpio for z in config.zones], config.pulse_width_s)
    drag_state = {
        "mode": None,  # None | "move" | "resize"
        "zone_name": None,
        "offset_x": 0,
        "offset_y": 0,
        "start_x": 0,
        "start_y": 0,
        "orig_x": 0,
        "orig_y": 0,
        "orig_w": 0,
        "orig_h": 0,
    }
    resize_handle_px = 14
    min_zone_size = 12

    def rebuild_gpio_if_needed():
        nonlocal gpio
        pins = [z.gpio for z in config.zones]
        if gpio.has_same_pins(pins):
            return
        with gpio_lock:
            gpio.close()
            gpio = GpioManager(pins, config.pulse_width_s)

    RuntimeControlPanel(config, runtime, gray0.shape, rebuild_gpio_if_needed).start()

    def reinit_zone_background(zone: RoiZone) -> None:
        if runtime.latest_gray is None:
            return
        roi = runtime.latest_gray[zone.y : zone.y + zone.h, zone.x : zone.x + zone.w]
        if roi.size == 0:
            return
        zone.background = cv2.GaussianBlur(roi, (config.blur_kernel, config.blur_kernel), 0).astype(np.float32)
        zone.consecutive_count = 0
        zone.last_trigger_time = 0.0

    def on_lab_mouse(event, x, y, _flags, _param):
        with runtime.lock:
            zones = config.zones
            if event == cv2.EVENT_LBUTTONDOWN:
                for zone in reversed(zones):
                    in_zone = zone.x <= x <= zone.x + zone.w and zone.y <= y <= zone.y + zone.h
                    if not in_zone:
                        continue
                    near_corner = (
                        abs(x - (zone.x + zone.w)) <= resize_handle_px
                        and abs(y - (zone.y + zone.h)) <= resize_handle_px
                    )
                    drag_state["zone_name"] = zone.name
                    if near_corner:
                        drag_state["mode"] = "resize"
                        drag_state["start_x"] = x
                        drag_state["start_y"] = y
                        drag_state["orig_w"] = zone.w
                        drag_state["orig_h"] = zone.h
                    else:
                        drag_state["mode"] = "move"
                        drag_state["offset_x"] = x - zone.x
                        drag_state["offset_y"] = y - zone.y
                    drag_state["orig_x"] = zone.x
                    drag_state["orig_y"] = zone.y
                    break

            elif event == cv2.EVENT_MOUSEMOVE and drag_state["mode"] is not None:
                zone = next((z for z in zones if z.name == drag_state["zone_name"]), None)
                if zone is None:
                    drag_state["mode"] = None
                    return
                if drag_state["mode"] == "move":
                    new_x = x - drag_state["offset_x"]
                    new_y = y - drag_state["offset_y"]
                    zone.x = max(0, min(new_x, config.frame_width - zone.w))
                    zone.y = max(0, min(new_y, config.frame_height - zone.h))
                else:
                    dx = x - drag_state["start_x"]
                    dy = y - drag_state["start_y"]
                    zone.w = max(min_zone_size, min(drag_state["orig_w"] + dx, config.frame_width - zone.x))
                    zone.h = max(min_zone_size, min(drag_state["orig_h"] + dy, config.frame_height - zone.y))

            elif event == cv2.EVENT_LBUTTONUP and drag_state["mode"] is not None:
                zone = next((z for z in zones if z.name == drag_state["zone_name"]), None)
                if zone is not None:
                    moved_or_resized = (
                        zone.x != drag_state["orig_x"]
                        or zone.y != drag_state["orig_y"]
                        or zone.w != drag_state["orig_w"]
                        or zone.h != drag_state["orig_h"]
                    )
                    if moved_or_resized:
                        reinit_zone_background(zone)
                drag_state["mode"] = None
                drag_state["zone_name"] = None

    try:
        cv2.namedWindow("Lab Feed")
        cv2.setMouseCallback("Lab Feed", on_lab_mouse)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            with runtime.lock:
                runtime.latest_gray = gray.copy()
                zones = list(config.zones)
                cfg = MazeConfig(**{**config.__dict__, "zones": zones})
            combined_mask = np.zeros_like(gray)
            active: dict[str, bool] = {}
            debug: dict[str, str] = {}
            now = time.time()
            for zone in zones:
                if zone.background is None:
                    init_zone_runtime([zone], gray, cfg.blur_kernel)
                raw_motion, mask, amount = process_zone(zone, gray, cfg)
                zone.consecutive_count = zone.consecutive_count + 1 if raw_motion else 0
                cd_left = max(0.0, cfg.cooldown_s - (now - zone.last_trigger_time))
                if zone.consecutive_count >= cfg.min_consecutive_frames and cd_left <= 0:
                    with gpio_lock:
                        gpio.pulse_async(zone.gpio)
                    zone.last_trigger_time = now
                    zone.consecutive_count = 0
                active[zone.name] = raw_motion
                debug[zone.name] = f"amt:{amount:.0f} cnt:{zone.consecutive_count} cd:{cd_left:.1f}s"
                combined_mask[zone.y : zone.y + zone.h, zone.x : zone.x + zone.w] = cv2.bitwise_or(combined_mask[zone.y : zone.y + zone.h, zone.x : zone.x + zone.w], mask)
            with runtime.lock:
                zone_by_name = {z.name: z for z in zones}
                for i, z in enumerate(config.zones):
                    if z.name in zone_by_name:
                        config.zones[i] = zone_by_name[z.name]
            draw_zones(frame, zones, active, debug)
            cv2.putText(
                frame,
                "Q=quit | Drag zone to move | Drag bottom-right corner to resize",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow("Lab Feed", frame)
            cv2.imshow("Motion Mask", combined_mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        runtime.running = False
        with gpio_lock:
            gpio.close()
        cv2.destroyAllWindows()


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        root = tk.Tk(); root.withdraw(); messagebox.showerror("Camera", "Could not open camera"); root.destroy(); return
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        root = tk.Tk(); root.withdraw(); messagebox.showerror("Camera", "Could not read frame from camera"); root.destroy(); return
    try:
        setup = MazeSetupGUI(first_frame)
        result = setup.run()
    except RuntimeError as exc:
        cap.release()
        root = tk.Tk(); root.withdraw(); messagebox.showerror("Error", str(exc)); root.destroy(); return
    if result is None:
        cap.release(); return
    config, _ = result
    cap.release()
    cap = open_camera(config)
    if not cap.isOpened():
        root = tk.Tk(); root.withdraw(); messagebox.showerror("Camera", "Could not open camera for monitoring"); root.destroy(); return
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        root = tk.Tk(); root.withdraw(); messagebox.showerror("Camera", "Could not read frame for monitoring"); root.destroy(); return
    validate_config(config, frame_width=first_frame.shape[1], frame_height=first_frame.shape[0])
    try:
        run_monitoring(cap, config, first_frame)
    finally:
        cap.release()
