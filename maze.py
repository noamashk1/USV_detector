import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    import lgpio
except ImportError:
    lgpio = None

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "maze_config.json"

DEFAULT_CONFIG = {
    "frame_width": 640,
    "frame_height": 480,
    "motion_threshold": 500,
    "pulse_width_s": 0.05,
    "diff_threshold": 25,
    "blur_kernel": 21,
    "zones": [],
}

ZONE_COLORS = [
    (0, 255, 0),
    (255, 128, 0),
    (0, 200, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 255, 128),
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
        json.dump(config.to_dict(), f, indent=2)
    print(f"Config saved to {path}")


def validate_config(config: MazeConfig, frame_width: int | None = None, frame_height: int | None = None) -> None:
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
            print("lgpio not available — TTL output disabled")
            return

        self._handle = lgpio.gpiochip_open(0)
        for pin in self._pins:
            err = lgpio.gpio_claim_output(self._handle, pin)
            if err < 0:
                raise RuntimeError(f"lgpio gpio_claim_output failed for GPIO{pin}: {err}")
            lgpio.gpio_write(self._handle, pin, 0)
        self._available = True
        print(f"TTL output enabled on GPIO pins: {self._pins}")

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


def choose_startup_mode() -> str:
    print("\nMaze motion detector — setup")
    print("  [1] Load config from JSON")
    print("  [2] Interactive setup (save JSON when done)")
    while True:
        choice = input("Choose mode [1/2]: ").strip()
        if choice in ("1", "2"):
            return choice
        print("Invalid choice, enter 1 or 2.")


def load_config_interactive() -> tuple[MazeConfig, Path]:
    default = str(DEFAULT_CONFIG_PATH)
    path_str = input(f"Config path [{default}]: ").strip() or default
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return load_config(path), path


def setup_zones_interactive(first_frame: np.ndarray, config: MazeConfig) -> MazeConfig:
    fh, fw = first_frame.shape[:2]
    config.frame_width = fw
    config.frame_height = fh
    config.zones = []

    print("\nInteractive zone setup")
    print("Select each ROI with the mouse, then press ENTER or SPACE.")
    zone_idx = 1
    while True:
        title = f"Select ROI for zone {zone_idx}"
        r = cv2.selectROI(title, first_frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(title)
        x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        if w == 0 or h == 0:
            if not config.zones:
                print("No zone selected. Try again.")
                continue
            break

        default_name = f"zone_{zone_idx}"
        name = input(f"Zone name [{default_name}]: ").strip() or default_name
        while True:
            gpio_str = input("GPIO pin (BCM): ").strip()
            try:
                gpio = int(gpio_str)
                break
            except ValueError:
                print("Enter a valid integer GPIO number.")

        zone = RoiZone(name=name, x=x, y=y, w=w, h=h, gpio=gpio)
        config.zones.append(zone)
        validate_config(config, frame_width=fw, frame_height=fh)

        more = input("Add another zone? [y/N]: ").strip().lower()
        if more not in ("y", "yes"):
            break
        zone_idx += 1

    return config


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


def main() -> None:
    mode = choose_startup_mode()
    config_path = DEFAULT_CONFIG_PATH

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        return

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        print("Camera error: could not read first frame")
        return

    actual_h, actual_w = first_frame.shape[:2]

    if mode == "1":
        cap.release()
        try:
            config, config_path = load_config_interactive()
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            print(f"Failed to load config: {exc}")
            return
        cap = open_camera(config)
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            print("Camera error after reopen")
            return
        actual_h, actual_w = first_frame.shape[:2]
        validate_config(config, frame_width=actual_w, frame_height=actual_h)
    else:
        config = MazeConfig.from_dict(DEFAULT_CONFIG)
        config.frame_width = actual_w
        config.frame_height = actual_h
        try:
            config = setup_zones_interactive(first_frame, config)
        except ValueError as exc:
            cap.release()
            print(f"Setup error: {exc}")
            return

        save_path_str = input(f"Save config to [{config_path}]: ").strip()
        config_path = Path(save_path_str) if save_path_str else config_path
        save_config(config, config_path)

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    init_baselines(config.zones, first_gray, config.blur_kernel)

    gpio = GpioManager(
        pins=[z.gpio for z in config.zones],
        pulse_width_s=config.pulse_width_s,
    )

    print(f"\nMonitoring {len(config.zones)} zone(s). Press 'q' to stop.")
    for zone in config.zones:
        print(f"  - {zone.name}: ROI ({zone.x},{zone.y}) {zone.w}x{zone.h} -> GPIO{zone.gpio}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
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
            cv2.imshow("Lab Feed", frame)
            cv2.imshow("Motion Mask", combined_mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        gpio.close()


if __name__ == "__main__":
    main()
