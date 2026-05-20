import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import numpy as np
from datetime import datetime

script_dir = pathlib.Path(__file__).resolve().parent
LIDAR_ROOT = script_dir / "lidar with measurements"

if not LIDAR_ROOT.exists():
    raise FileNotFoundError(f"LIDAR root does not exist: {LIDAR_ROOT}")


def plot_folder(folder_path: pathlib.Path) -> None:
    csv_files = sorted(folder_path.glob("*.csv")) + sorted(folder_path.glob("*.CSV"))
    if not csv_files:
        return

    csv_files = csv_files[-50:]
    print(f"Processing {len(csv_files)} CSV files in {folder_path}")

    plt.figure(figsize=(14, 14))

    for csv_path in csv_files:
        try:
            data = pd.read_csv(csv_path, header=None, usecols=[0, 1, 2])
        except Exception as exc:
            print(f"Skipping unreadable file {csv_path}: {exc}")
            continue

        if data.shape[1] != 3:
            print(f"Skipping malformed file (wrong column count): {csv_path}")
            continue

        data.columns = ['distance_mm', 'angle_deg', 'quality']

        # ---- ALL COMPUTATION MUST BE INSIDE LOOP ----
        distance_m = data['distance_mm'] / 1000.0
        angles_rad = np.radians(data['angle_deg'])

        x = distance_m * np.cos(angles_rad)
        y = distance_m * np.sin(angles_rad)

        quality = data['quality'].astype(float)
        quality_norm = (quality - quality.min()) / (quality.max() - quality.min() + 1e-9)

        plt.scatter(
            x,
            y,
            s=5,
            c='black',
            alpha=quality_norm
        )

    # ---- FIGURE STYLING (OUTSIDE LOOP) ----
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title(f'2D LIDAR Scans ({len(csv_files)} files)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = folder_path / f"lidar_plot_{timestamp}.png"

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {filename}")


def main() -> None:
    folders = [LIDAR_ROOT] + sorted([p for p in LIDAR_ROOT.rglob("*") if p.is_dir()])
    for folder in folders:
        plot_folder(folder)


if __name__ == "__main__":
    main()