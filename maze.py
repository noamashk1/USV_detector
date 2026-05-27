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

ZONE_COLORS_RGB = [
    "#00ff00",
    "#ff8000",
    "#00c8ff",
    "#ff00ff",
    "#ffff00",
    "#80ff80",
]


@dataclass
class RoiZone:
    name: str
    x: int
    y: int
    w: int
    h: int
    gpio: int
    baseline: np.ndarray | None = field(default=None, repr=False)
    prev_motion: bool = False

    def rect(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "gpio": self.gpio,
        }

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
    zones: list[RoiZone] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "motion_threshold": self.motion_threshold,
            "pulse_width_s": self.pulse_width_s,
            "diff_threshold": self.diff_threshold,
            "blur_kernel": self.blur_kernel,
            "zones": [z.to_dict() for z in self.zones],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MazeConfig":
        zones = [RoiZone.from_dict(z) for z in data.get("zones", [])]
        return cls(
            frame_width=int(data.get("frame_width", 640)),
            frame_height=int(data.get("frame_height", 480)),
            motion_threshold=int(data.get("motion_threshold", 500)),
            pulse_width_s=float(data.get("pulse_width_s", 0.05)),
            diff_threshold=int(data.get("diff_threshold", 25)),
            blur_kernel=int(data.get("blur_kernel", 21)),
            zones=zones,
        )


def load_config(path: Path) -> MazeConfig:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    config = MazeConfig.from_dict(data)
    validate_config(config)
    return config


def save_config(config: MazeConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)


def validate_config(
    config: MazeConfig,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> None:
    if not config.zones:
        raise ValueError("Config must contain at least one zone")

    if config.blur_kernel % 2 == 0 or config.blur_kernel < 3:
        raise ValueError("blur_kernel must be an odd integer >= 3")

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

        if not self._pins:
            return
        if lgpio is None:
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
        self._handle = None
        self._available = False


class MazeSetupGUI:
    """All setup via GUI (no console input)."""

    def __init__(self, first_frame: np.ndarray):
        if ImageTk is None:
            raise RuntimeError(
                "Pillow is required for image display in the GUI.\nInstall: pip install Pillow"
            )

        self._frame_bgr = first_frame.copy()
        fh, fw = first_frame.shape[:2]
        self.config = MazeConfig(frame_width=fw, frame_height=fh)
        self.config_path = DEFAULT_CONFIG_PATH
        self.result: tuple[MazeConfig, Path] | None = None
        self._save_on_start = False

        self._scale = 1.0
        self._photo: ImageTk.PhotoImage | None = None
        self._drag_start: tuple[int, int] | None = None
        self._pending_roi: tuple[int, int, int, int] | None = None
        self._rect_item: int | None = None
        self._zone_items: list[int] = []

        self.root = tk.Tk()
        self.root.title("Maze — Setup")
        self.root.option_add("*Font", "TkDefaultFont")
        self._build_ui()
        self._show_mode_screen()

    def _build_ui(self) -> None:
        self.container = ttk.Frame(self.root, padding=12)
        self.container.pack(fill=tk.BOTH, expand=True)

        self.mode_frame = ttk.Frame(self.container)
        self.editor_frame = ttk.Frame(self.container)

        # --- Mode selection screen ---
        ttk.Label(self.mode_frame, text="Maze Motion Detector", font=("", 14, "bold")).pack(
            anchor="w", pady=(0, 16)
        )
        ttk.Label(self.mode_frame, text="Choose setup mode:").pack(anchor="w", pady=(0, 8))

        btn_row = ttk.Frame(self.mode_frame)
        btn_row.pack(fill=tk.X, pady=8)
        ttk.Button(
            btn_row,
            text="Load settings from JSON",
            command=self._on_load_json,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            btn_row,
            text="Interactive setup",
            command=self._on_interactive,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Exit", command=self._cancel).pack(side=tk.RIGHT, padx=4)

        # --- Editor screen ---
        top = ttk.Frame(self.editor_frame)
        top.pack(fill=tk.BOTH, expand=True)

        canvas_frame = ttk.LabelFrame(top, text="Camera — drag a rectangle to select ROI", padding=4)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        self.canvas = tk.Canvas(canvas_frame, bg="#222222", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        side = ttk.Frame(top, width=280)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        ttk.Label(side, text="Global parameters", font=("", 11, "bold")).pack(anchor="w", pady=(0, 6))
        self._add_param_row(side, "Motion threshold:", "motion_threshold", "500")
        self._add_param_row(side, "Pulse width (s):", "pulse_width_s", "0.05")
        self._add_param_row(side, "Diff threshold:", "diff_threshold", "25")
        self._add_param_row(side, "Blur kernel:", "blur_kernel", "21")

        zone_box = ttk.LabelFrame(side, text="New zone", padding=6)
        zone_box.pack(fill=tk.X, pady=10)
        self.name_var = tk.StringVar(value="zone_1")
        self.gpio_var = tk.StringVar(value="17")
        ttk.Label(zone_box, text="Zone name:").pack(anchor="w")
        ttk.Entry(zone_box, textvariable=self.name_var, width=20).pack(anchor="w", fill=tk.X, pady=2)
        ttk.Label(zone_box, text="GPIO pin (BCM):").pack(anchor="w", pady=(6, 0))
        ttk.Entry(zone_box, textvariable=self.gpio_var, width=20).pack(anchor="w", fill=tk.X, pady=2)
        ttk.Button(zone_box, text="Add zone from selection", command=self._add_zone).pack(
            anchor="w", pady=(8, 0)
        )

        list_box = ttk.LabelFrame(side, text="Configured zones", padding=6)
        list_box.pack(fill=tk.BOTH, expand=True, pady=6)
        self.zone_list = tk.Listbox(list_box, height=8)
        self.zone_list.pack(fill=tk.BOTH, expand=True)
        ttk.Button(list_box, text="Remove selected zone", command=self._remove_zone).pack(anchor="w", pady=4)

        action_row = ttk.Frame(self.editor_frame)
        action_row.pack(fill=tk.X, pady=(12, 0))
        self.status_var = tk.StringVar(value="")
        ttk.Label(action_row, textvariable=self.status_var, foreground="#333").pack(
            anchor="w", fill=tk.X
        )
        btns = ttk.Frame(action_row)
        btns.pack(fill=tk.X, pady=6)
        ttk.Button(btns, text="Start monitoring", command=self._on_start).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Save JSON...", command=self._save_as).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Back", command=self._show_mode_screen).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Exit", command=self._cancel).pack(side=tk.RIGHT, padx=4)

    def _add_param_row(self, parent: ttk.Frame, label: str, attr: str, default: str) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
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
        self.root.geometry("480x200")

    def _show_editor_screen(self) -> None:
        self.mode_frame.pack_forget()
        self.editor_frame.pack(fill=tk.BOTH, expand=True)
        self.root.geometry("1000x700")
        self._render_canvas()
        self._refresh_zone_list()

    def _on_load_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Select settings file",
            initialdir=str(SCRIPT_DIR),
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
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
        fh, fw = self._frame_bgr.shape[:2]
        self.config = MazeConfig(frame_width=fw, frame_height=fh)
        self.config.zones = []
        self.config_path = DEFAULT_CONFIG_PATH
        self._save_on_start = True
        self._sync_params_to_ui()
        self.name_var.set("zone_1")
        self.gpio_var.set("17")
        self._show_editor_screen()
        self.status_var.set("Drag a rectangle, enter name and GPIO, then click 'Add zone from selection'")

    def _sync_params_to_ui(self) -> None:
        self.motion_threshold_var.set(str(self.config.motion_threshold))
        self.pulse_width_s_var.set(str(self.config.pulse_width_s))
        self.diff_threshold_var.set(str(self.config.diff_threshold))
        self.blur_kernel_var.set(str(self.config.blur_kernel))

    def _apply_params_from_ui(self) -> None:
        try:
            self.config.motion_threshold = int(self.motion_threshold_var.get())
            self.config.pulse_width_s = float(self.pulse_width_s_var.get())
            self.config.diff_threshold = int(self.diff_threshold_var.get())
            self.config.blur_kernel = int(self.blur_kernel_var.get())
        except ValueError as exc:
            raise ValueError("Invalid global parameter — values must be numeric") from exc

    def _render_canvas(self) -> None:
        h, w = self._frame_bgr.shape[:2]
        max_w, max_h = 720, 540
        self._scale = min(max_w / w, max_h / h, 1.0)
        disp_w = max(1, int(w * self._scale))
        disp_h = max(1, int(h * self._scale))
        resized = cv2.resize(self._frame_bgr, (disp_w, disp_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
        self._zone_items.clear()
        self._rect_item = None
        self._pending_roi = None
        for i, zone in enumerate(self.config.zones):
            self._draw_zone_rect(zone, i)

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
        self._rect_item = self.canvas.create_rectangle(
            x0, y0, event.x, event.y, outline="#ffff00", width=2
        )

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
            self.status_var.set(f"Selected ROI: ({fx0},{fy0}) {w}x{h} — click 'Add zone from selection'")
        else:
            self._pending_roi = None
            self.status_var.set("Selection too small — try again")

    def _draw_zone_rect(self, zone: RoiZone, index: int) -> None:
        x = int(zone.x * self._scale)
        y = int(zone.y * self._scale)
        w = int(zone.w * self._scale)
        h = int(zone.h * self._scale)
        color = ZONE_COLORS_RGB[index % len(ZONE_COLORS_RGB)]
        rid = self.canvas.create_rectangle(x, y, x + w, y + h, outline=color, width=2)
        self._zone_items.append(rid)
        self.canvas.create_text(
            x + 4, y + 14, anchor=tk.NW, text=f"{zone.name} GPIO{zone.gpio}", fill=color
        )

    def _refresh_zone_list(self) -> None:
        self.zone_list.delete(0, tk.END)
        for z in self.config.zones:
            self.zone_list.insert(
                tk.END, f"{z.name}  |  ({z.x},{z.y}) {z.w}x{z.h}  |  GPIO{z.gpio}"
            )

    def _add_zone(self) -> None:
        if self._pending_roi is None:
            messagebox.showwarning(
                "No selection",
                "Drag a rectangle on the image before adding a zone.",
                parent=self.root,
            )
            return
        name = self.name_var.get().strip() or f"zone_{len(self.config.zones) + 1}"
        try:
            gpio = int(self.gpio_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid GPIO", "Enter an integer GPIO pin number.", parent=self.root)
            return

        x, y, w, h = self._pending_roi
        zone = RoiZone(name=name, x=x, y=y, w=w, h=h, gpio=gpio)
        try:
            self._apply_params_from_ui()
            trial = MazeConfig(
                frame_width=self.config.frame_width,
                frame_height=self.config.frame_height,
                motion_threshold=self.config.motion_threshold,
                pulse_width_s=self.config.pulse_width_s,
                diff_threshold=self.config.diff_threshold,
                blur_kernel=self.config.blur_kernel,
                zones=self.config.zones + [zone],
            )
            validate_config(
                trial,
                frame_width=self.config.frame_width,
                frame_height=self.config.frame_height,
            )
        except ValueError as exc:
            messagebox.showerror("Error", str(exc), parent=self.root)
            return

        self.config.zones.append(zone)
        self._pending_roi = None
        if self._rect_item is not None:
            self.canvas.delete(self._rect_item)
            self._rect_item = None
        self._render_canvas()
        self._refresh_zone_list()
        self.name_var.set(f"zone_{len(self.config.zones) + 1}")
        self.status_var.set(f"Added zone '{name}' (GPIO{gpio})")

    def _remove_zone(self) -> None:
        sel = self.zone_list.curselection()
        if not sel:
            messagebox.showwarning(
                "Nothing selected", "Select a zone from the list to remove.", parent=self.root
            )
            return
        idx = sel[0]
        del self.config.zones[idx]
        self._render_canvas()
        self._refresh_zone_list()
        self.status_var.set("Zone removed")

    def _save_as(self) -> None:
        try:
            self._apply_params_from_ui()
            validate_config(
                self.config,
                frame_width=self.config.frame_width,
                frame_height=self.config.frame_height,
            )
        except ValueError as exc:
            messagebox.showerror("Error", str(exc), parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            title="Save settings",
            initialdir=str(SCRIPT_DIR),
            initialfile="maze_config.json",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            save_config(self.config, Path(path))
            self.config_path = Path(path)
            self.status_var.set(f"Saved: {path}")
            messagebox.showinfo("Saved", f"Settings saved to:\n{path}", parent=self.root)
        except OSError as exc:
            messagebox.showerror("Save error", str(exc), parent=self.root)

    def _on_start(self) -> None:
        try:
            self._apply_params_from_ui()
            validate_config(
                self.config,
                frame_width=self.config.frame_width,
                frame_height=self.config.frame_height,
            )
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


def init_baselines(zones: list[RoiZone], gray_frame: np.ndarray, blur_kernel: int) -> None:
    k = (blur_kernel, blur_kernel)
    for zone in zones:
        x, y, w, h = zone.rect()
        roi = gray_frame[y : y + h, x : x + w]
        zone.baseline = cv2.GaussianBlur(roi, k, 0)
        zone.prev_motion = False


def process_zone(
    zone: RoiZone,
    gray: np.ndarray,
    blur_kernel: int,
    diff_threshold: int,
    motion_threshold: int,
) -> tuple[bool, np.ndarray]:
    x, y, w, h = zone.rect()
    k = (blur_kernel, blur_kernel)
    roi = gray[y : y + h, x : x + w]
    roi_blurred = cv2.GaussianBlur(roi, k, 0)
    frame_delta = cv2.absdiff(zone.baseline, roi_blurred)
    thresh = cv2.threshold(frame_delta, diff_threshold, 255, cv2.THRESH_BINARY)[1]
    motion_amount = np.sum(thresh) / 255
    motion_detected = motion_amount > motion_threshold
    return motion_detected, thresh


def draw_zones(frame: np.ndarray, zones: list[RoiZone], active: dict[str, bool]) -> None:
    for i, zone in enumerate(zones):
        x, y, w, h = zone.rect()
        color = ZONE_COLORS[i % len(ZONE_COLORS)]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{zone.name} GPIO{zone.gpio}"
        cv2.putText(frame, label, (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if active.get(zone.name):
            cv2.putText(
                frame,
                "MOTION",
                (x + 4, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )


def open_camera(config: MazeConfig) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
    return cap


def run_monitoring(cap: cv2.VideoCapture, config: MazeConfig, first_frame: np.ndarray) -> None:
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    init_baselines(config.zones, first_gray, config.blur_kernel)

    gpio: GpioManager | None = None
    try:
        gpio = GpioManager(
            pins=[z.gpio for z in config.zones],
            pulse_width_s=config.pulse_width_s,
        )
    except RuntimeError as exc:
        messagebox.showerror("GPIO error", str(exc))
        return

    if gpio is not None and not gpio.available and lgpio is None:
        messagebox.showwarning(
            "GPIO unavailable",
            "lgpio is not installed — monitoring will continue without TTL output.",
        )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Camera", "Failed to read frame from camera")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            combined_mask = np.zeros_like(gray)
            active: dict[str, bool] = {}

            for zone in config.zones:
                motion_detected, thresh = process_zone(
                    zone,
                    gray,
                    config.blur_kernel,
                    config.diff_threshold,
                    config.motion_threshold,
                )
                active[zone.name] = motion_detected

                if motion_detected and not zone.prev_motion:
                    gpio.pulse_async(zone.gpio)

                zone.prev_motion = motion_detected

                x, y, w, h = zone.rect()
                combined_mask[y : y + h, x : x + w] = cv2.bitwise_or(
                    combined_mask[y : y + h, x : x + w],
                    thresh,
                )

            draw_zones(frame, config.zones, active)
            cv2.putText(
                frame,
                "Press Q to stop",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Lab Feed", frame)
            cv2.imshow("Motion Mask", combined_mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if gpio is not None:
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

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Camera", "Could not read frame from camera")
        root.destroy()
        return

    try:
        setup = MazeSetupGUI(first_frame)
        result = setup.run()
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

    config, _config_path = result
    cap.release()

    cap = open_camera(config)
    if not cap.isOpened():
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Camera", "Could not open camera for monitoring")
        root.destroy()
        return

    ret, first_frame = cap.read()
    if not ret:
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


if __name__ == "__main__":
    main()
