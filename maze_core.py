import json
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

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

ZONE_COLORS = [
    (0, 255, 0),
    (255, 128, 0),
    (0, 200, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 255, 128),
]
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
        return cls(
            name=str(data["name"]),
            x=int(data["x"]),
            y=int(data["y"]),
            w=int(data["w"]),
            h=int(data["h"]),
            gpio=int(data["gpio"]),
        )


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
        self._photo: ImageTk.PhotoImage | None = None
        self._drag_start: tuple[int, int] | None = None
        self._pending_roi: tuple[int, int, int, int] | None = None
        self._rect_item: int | None = None

        self.root = tk.Tk()
        self.root.title("Maze — Setup")
        self._build_ui()
        self._show_mode_screen()

    def _build_ui(self) -> None:
        self.container = ttk.Frame(self.root, padding=12)
        self.container.pack(fill=tk.BOTH, expand=True)
        self.mode_frame = ttk.Frame(self.container)
        self.editor_frame = ttk.Frame(self.container)

        ttk.Label(self.mode_frame, text="Maze Motion Detector", font=("", 14, "bold")).pack(anchor="w", pady=(0, 16))
        ttk.Label(self.mode_frame, text="Choose setup mode:").pack(anchor="w", pady=(0, 8))
        r = ttk.Frame(self.mode_frame)
        r.pack(fill=tk.X, pady=8)
        ttk.Button(r, text="Load settings from JSON", command=self._on_load_json).pack(side=tk.LEFT, padx=4)
        ttk.Button(r, text="Interactive setup", command=self._on_interactive).pack(side=tk.LEFT, padx=4)
        ttk.Button(r, text="Exit", command=self._cancel).pack(side=tk.RIGHT, padx=4)

        top = ttk.Frame(self.editor_frame)
        top.pack(fill=tk.BOTH, expand=True)
        canvas_frame = ttk.LabelFrame(top, text="Camera — drag rectangle to select ROI", padding=4)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        self.canvas = tk.Canvas(canvas_frame, bg="#222", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        side = ttk.Frame(top, width=320)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)
        ttk.Label(side, text="Global parameters", font=("", 11, "bold")).pack(anchor="w", pady=(0, 6))
        self._add_param_row(side, "Motion threshold:", "motion_threshold", "500")
        self._add_param_row(side, "Pulse width (s):", "pulse_width_s", "0.05")
        self._add_param_row(side, "Diff threshold:", "diff_threshold", "25")
        self._add_param_row(side, "Blur kernel:", "blur_kernel", "21")
        self._add_param_row(side, "Background alpha:", "background_alpha", "0.02")
        self._add_param_row(side, "Min consecutive frames:", "min_consecutive_frames", "2")
        self._add_param_row(side, "Cooldown (s):", "cooldown_s", "0.3")
        self._add_param_row(side, "Morph kernel size:", "morph_kernel_size", "3")
        self._add_param_row(side, "Morph open iterations:", "morph_open_iterations", "1")
        self._add_param_row(side, "Morph close iterations:", "morph_close_iterations", "1")
        self._add_param_row(side, "Adaptive k sigma:", "adaptive_k_sigma", "2.5")
        self.use_adaptive_threshold_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="Use adaptive threshold", variable=self.use_adaptive_threshold_var).pack(anchor="w", pady=(2, 8))

        zone_box = ttk.LabelFrame(side, text="New zone", padding=6)
        zone_box.pack(fill=tk.X, pady=8)
        self.name_var = tk.StringVar(value="zone_1")
        self.gpio_var = tk.StringVar(value="17")
        ttk.Label(zone_box, text="Zone name:").pack(anchor="w")
        ttk.Entry(zone_box, textvariable=self.name_var, width=20).pack(anchor="w", fill=tk.X, pady=2)
        ttk.Label(zone_box, text="GPIO pin (BCM):").pack(anchor="w", pady=(6, 0))
        ttk.Entry(zone_box, textvariable=self.gpio_var, width=20).pack(anchor="w", fill=tk.X, pady=2)
        ttk.Button(zone_box, text="Add zone from selection", command=self._add_zone).pack(anchor="w", pady=(8, 0))

        list_box = ttk.LabelFrame(side, text="Configured zones", padding=6)
        list_box.pack(fill=tk.BOTH, expand=True, pady=6)
        self.zone_list = tk.Listbox(list_box, height=8)
        self.zone_list.pack(fill=tk.BOTH, expand=True)
        ttk.Button(list_box, text="Remove selected zone", command=self._remove_zone).pack(anchor="w", pady=4)

        action = ttk.Frame(self.editor_frame)
        action.pack(fill=tk.X, pady=(12, 0))
        self.status_var = tk.StringVar(value="")
        ttk.Label(action, textvariable=self.status_var, foreground="#333").pack(anchor="w", fill=tk.X)
        b = ttk.Frame(action)
        b.pack(fill=tk.X, pady=6)
        ttk.Button(b, text="Start monitoring", command=self._on_start).pack(side=tk.LEFT, padx=4)
        ttk.Button(b, text="Save JSON...", command=self._save_as).pack(side=tk.LEFT, padx=4)
        ttk.Button(b, text="Back", command=self._show_mode_screen).pack(side=tk.LEFT, padx=4)
        ttk.Button(b, text="Exit", command=self._cancel).pack(side=tk.RIGHT, padx=4)

    def _add_param_row(self, parent: ttk.Frame, label: str, attr: str, default: str) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        var = tk.StringVar(value=default)
        setattr(self, f"{attr}_var", var)
        ttk.Label(row, text=label).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var, width=10).pack(side=tk.LEFT, padx=4)

    def run(self) -> tuple[MazeConfig, Path] | None:
        self.root.protocol("WM_DELETE_WINDOW", self._cancel)
        self.root.mainloop()
        return self.result

    def _show_mode_screen(self) -> None:
        self.editor_frame.pack_forget()
        self.mode_frame.pack(fill=tk.BOTH, expand=True)
        self.root.geometry("500x220")

    def _show_editor_screen(self) -> None:
        self.mode_frame.pack_forget()
        self.editor_frame.pack(fill=tk.BOTH, expand=True)
        self.root.geometry("1180x760")
        self._render_canvas()
        self._refresh_zone_list()

    def _on_load_json(self) -> None:
        path = filedialog.askopenfilename(title="Select settings file", initialdir=str(SCRIPT_DIR), filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.config = load_config(Path(path))
            self.config_path = Path(path)
            self._save_on_start = False
            self._sync_params_to_ui()
            self._show_editor_screen()
            self.status_var.set(f"Loaded: {path}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Load error", str(exc), parent=self.root)

    def _on_interactive(self) -> None:
        h, w = self._frame_bgr.shape[:2]
        self.config = MazeConfig(frame_width=w, frame_height=h)
        self.config.zones = []
        self.config_path = DEFAULT_CONFIG_PATH
        self._save_on_start = True
        self._sync_params_to_ui()
        self.name_var.set("zone_1")
        self.gpio_var.set("17")
        self._show_editor_screen()
        self.status_var.set("Drag ROI, set name/GPIO, then click 'Add zone from selection'")

    def _sync_params_to_ui(self) -> None:
        for attr in [
            "motion_threshold",
            "pulse_width_s",
            "diff_threshold",
            "blur_kernel",
            "background_alpha",
            "min_consecutive_frames",
            "cooldown_s",
            "morph_kernel_size",
            "morph_open_iterations",
            "morph_close_iterations",
            "adaptive_k_sigma",
        ]:
            getattr(self, f"{attr}_var").set(str(getattr(self.config, attr)))
        self.use_adaptive_threshold_var.set(bool(self.config.use_adaptive_threshold))

    def _apply_params_from_ui(self) -> None:
        try:
            self.config.motion_threshold = int(self.motion_threshold_var.get())
            self.config.pulse_width_s = float(self.pulse_width_s_var.get())
            self.config.diff_threshold = int(self.diff_threshold_var.get())
            self.config.blur_kernel = int(self.blur_kernel_var.get())
            self.config.background_alpha = float(self.background_alpha_var.get())
            self.config.min_consecutive_frames = int(self.min_consecutive_frames_var.get())
            self.config.cooldown_s = float(self.cooldown_s_var.get())
            self.config.morph_kernel_size = int(self.morph_kernel_size_var.get())
            self.config.morph_open_iterations = int(self.morph_open_iterations_var.get())
            self.config.morph_close_iterations = int(self.morph_close_iterations_var.get())
            self.config.adaptive_k_sigma = float(self.adaptive_k_sigma_var.get())
            self.config.use_adaptive_threshold = bool(self.use_adaptive_threshold_var.get())
        except ValueError as exc:
            raise ValueError("Invalid parameter type") from exc

    def _render_canvas(self) -> None:
        h, w = self._frame_bgr.shape[:2]
        self._scale = min(760 / w, 560 / h, 1.0)
        dw, dh = max(1, int(w * self._scale)), max(1, int(h * self._scale))
        rgb = cv2.cvtColor(cv2.resize(self._frame_bgr, (dw, dh)), cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.config(width=dw, height=dh)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
        self._rect_item = None
        self._pending_roi = None
        for i, z in enumerate(self.config.zones):
            self._draw_zone_rect(z, i)

    def _canvas_to_frame(self, cx: int, cy: int) -> tuple[int, int]:
        return int(cx / self._scale), int(cy / self._scale)

    def _on_press(self, event: tk.Event) -> None:
        self._drag_start = (event.x, event.y)
        if self._rect_item is not None:
            self.canvas.delete(self._rect_item)
            self._rect_item = None

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        if self._rect_item is not None:
            self.canvas.delete(self._rect_item)
        self._rect_item = self.canvas.create_rectangle(x0, y0, event.x, event.y, outline="#ffff00", width=2)

    def _on_release(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        self._drag_start = None
        fx0, fy0 = self._canvas_to_frame(min(x0, event.x), min(y0, event.y))
        fx1, fy1 = self._canvas_to_frame(max(x0, event.x), max(y0, event.y))
        w, h = fx1 - fx0, fy1 - fy0
        if w > 5 and h > 5:
            self._pending_roi = (fx0, fy0, w, h)
            self.status_var.set(f"Selected ROI: ({fx0},{fy0}) {w}x{h}")
        else:
            self._pending_roi = None
            self.status_var.set("Selection too small")

    def _draw_zone_rect(self, zone: RoiZone, index: int) -> None:
        x, y, w, h = int(zone.x * self._scale), int(zone.y * self._scale), int(zone.w * self._scale), int(zone.h * self._scale)
        color = ZONE_COLORS_RGB[index % len(ZONE_COLORS_RGB)]
        self.canvas.create_rectangle(x, y, x + w, y + h, outline=color, width=2)
        self.canvas.create_text(x + 4, y + 14, anchor=tk.NW, text=f"{zone.name} GPIO{zone.gpio}", fill=color)

    def _refresh_zone_list(self) -> None:
        self.zone_list.delete(0, tk.END)
        for z in self.config.zones:
            self.zone_list.insert(tk.END, f"{z.name} | ({z.x},{z.y}) {z.w}x{z.h} | GPIO{z.gpio}")

    def _add_zone(self) -> None:
        if self._pending_roi is None:
            messagebox.showwarning("No selection", "Drag ROI before adding a zone.", parent=self.root)
            return
        name = self.name_var.get().strip() or f"zone_{len(self.config.zones) + 1}"
        try:
            gpio = int(self.gpio_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid GPIO", "GPIO pin must be integer.", parent=self.root)
            return
        x, y, w, h = self._pending_roi
        zone = RoiZone(name=name, x=x, y=y, w=w, h=h, gpio=gpio)
        try:
            self._apply_params_from_ui()
            trial = MazeConfig(**{**self.config.__dict__, "zones": self.config.zones + [zone]})
            validate_config(trial, frame_width=self.config.frame_width, frame_height=self.config.frame_height)
        except ValueError as exc:
            messagebox.showerror("Error", str(exc), parent=self.root)
            return
        self.config.zones.append(zone)
        self._pending_roi = None
        self._render_canvas()
        self._refresh_zone_list()
        self.name_var.set(f"zone_{len(self.config.zones) + 1}")
        self.status_var.set(f"Added zone '{name}' (GPIO{gpio})")

    def _remove_zone(self) -> None:
        sel = self.zone_list.curselection()
        if not sel:
            messagebox.showwarning("Nothing selected", "Select a zone to remove.", parent=self.root)
            return
        del self.config.zones[sel[0]]
        self._render_canvas()
        self._refresh_zone_list()
        self.status_var.set("Zone removed")

    def _save_as(self) -> None:
        try:
            self._apply_params_from_ui()
            validate_config(self.config, frame_width=self.config.frame_width, frame_height=self.config.frame_height)
        except ValueError as exc:
            messagebox.showerror("Error", str(exc), parent=self.root)
            return
        path = filedialog.asksaveasfilename(title="Save settings", initialdir=str(SCRIPT_DIR), initialfile="maze_config.json", defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            save_config(self.config, Path(path))
            self.config_path = Path(path)
            self.status_var.set(f"Saved: {path}")
        except OSError as exc:
            messagebox.showerror("Save error", str(exc), parent=self.root)

    def _on_start(self) -> None:
        try:
            self._apply_params_from_ui()
            validate_config(self.config, frame_width=self.config.frame_width, frame_height=self.config.frame_height)
        except ValueError as exc:
            messagebox.showerror("Error", str(exc), parent=self.root)
            return
        if self._save_on_start:
            try:
                save_config(self.config, self.config_path)
            except OSError as exc:
                messagebox.showerror("Save error", str(exc), parent=self.root)
                return
        self.result = (self.config, self.config_path)
        self.root.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.root.destroy()


def init_zone_runtime(zones: list[RoiZone], gray_frame: np.ndarray, blur_kernel: int) -> None:
    k = (blur_kernel, blur_kernel)
    for zone in zones:
        x, y, w, h = zone.rect()
        zone.background = cv2.GaussianBlur(gray_frame[y : y + h, x : x + w], k, 0).astype(np.float32)
        zone.consecutive_count = 0
        zone.last_trigger_time = 0.0


def process_zone(zone: RoiZone, gray: np.ndarray, config: MazeConfig) -> tuple[bool, np.ndarray, float]:
    x, y, w, h = zone.rect()
    roi_blurred = cv2.GaussianBlur(gray[y : y + h, x : x + w], (config.blur_kernel, config.blur_kernel), 0)
    diff = cv2.absdiff(cv2.convertScaleAbs(zone.background), roi_blurred)
    if config.use_adaptive_threshold:
        threshold = max(float(config.diff_threshold), float(np.std(diff)) * config.adaptive_k_sigma)
    else:
        threshold = float(config.diff_threshold)
    mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((config.morph_kernel_size, config.morph_kernel_size), dtype=np.uint8)
    if config.morph_open_iterations > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=config.morph_open_iterations)
    if config.morph_close_iterations > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=config.morph_close_iterations)
    motion_amount = float(np.sum(mask) / 255.0)
    raw_motion = motion_amount > config.motion_threshold
    cv2.accumulateWeighted(roi_blurred.astype(np.float32), zone.background, config.background_alpha)
    return raw_motion, mask, motion_amount


def draw_zones(frame: np.ndarray, zones: list[RoiZone], active: dict[str, bool], debug: dict[str, str]) -> None:
    for i, zone in enumerate(zones):
        x, y, w, h = zone.rect()
        color = ZONE_COLORS[i % len(ZONE_COLORS)]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{zone.name} GPIO{zone.gpio}", (x, max(y - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if active.get(zone.name):
            cv2.putText(frame, "MOTION", (x + 4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, debug.get(zone.name, ""), (x + 4, min(y + h - 8, frame.shape[0] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def open_camera(config: MazeConfig) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
    return cap


def run_monitoring(cap: cv2.VideoCapture, config: MazeConfig, first_frame: np.ndarray) -> None:
    init_zone_runtime(config.zones, cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), config.blur_kernel)
    try:
        gpio = GpioManager([z.gpio for z in config.zones], config.pulse_width_s)
    except RuntimeError as exc:
        messagebox.showerror("GPIO error", str(exc))
        return

    if not gpio.available and lgpio is None:
        messagebox.showwarning("GPIO unavailable", "lgpio is not installed. Monitoring continues without TTL.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                messagebox.showerror("Camera", "Failed to read frame from camera")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            combined_mask = np.zeros_like(gray)
            active: dict[str, bool] = {}
            debug: dict[str, str] = {}
            now = time.time()
            for zone in config.zones:
                raw_motion, mask, amount = process_zone(zone, gray, config)
                zone.consecutive_count = zone.consecutive_count + 1 if raw_motion else 0
                cooldown_left = max(0.0, config.cooldown_s - (now - zone.last_trigger_time))
                should_trigger = zone.consecutive_count >= config.min_consecutive_frames and cooldown_left <= 0.0
                if should_trigger:
                    gpio.pulse_async(zone.gpio)
                    zone.last_trigger_time = now
                    zone.consecutive_count = 0
                active[zone.name] = raw_motion
                debug[zone.name] = f"amt:{amount:.0f} cnt:{zone.consecutive_count} cd:{cooldown_left:.1f}s"
                x, y, w, h = zone.rect()
                combined_mask[y : y + h, x : x + w] = cv2.bitwise_or(combined_mask[y : y + h, x : x + w], mask)

            draw_zones(frame, config.zones, active, debug)
            cv2.putText(frame, "Press Q to stop", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Lab Feed", frame)
            cv2.imshow("Motion Mask", combined_mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        gpio.close()
        cv2.destroyAllWindows()


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Camera", "Could not open camera")
        root.destroy()
        return
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Camera", "Could not read frame from camera")
        root.destroy()
        return

    try:
        gui = MazeSetupGUI(first_frame)
        result = gui.run()
    except RuntimeError as exc:
        cap.release()
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", str(exc))
        root.destroy()
        return

    if result is None:
        cap.release()
        return
    config, _ = result
    cap.release()
    cap = open_camera(config)
    if not cap.isOpened():
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Camera", "Could not open camera for monitoring")
        root.destroy()
        return
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Camera", "Could not read frame for monitoring")
        root.destroy()
        return
    try:
        validate_config(config, frame_width=first_frame.shape[1], frame_height=first_frame.shape[0])
    except ValueError as exc:
        cap.release()
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Config error", str(exc))
        root.destroy()
        return
    try:
        run_monitoring(cap, config, first_frame)
    finally:
        cap.release()
