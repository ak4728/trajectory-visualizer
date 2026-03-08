# trajectory-visualizer
DataFromSky trajectory visualizer

Visualize trajectories on top of a drone image and color trajectories by lane choice.

## Requirements

- Python 3.9+
- `matplotlib`

Install dependency:

```powershell
pip install matplotlib
```

## Usage

Basic overlay (colored by `Exit Gate`):

```powershell
python visualize_trajectories.py --csv trajectories.csv --image path\to\drone_frame.png
```

Save output image instead of opening a window:

```powershell
python visualize_trajectories.py --csv trajectories.csv --image path\to\drone_frame.png --output output\overlay.png
```

Coloring options for lane selection:

- `--lane-by exit_gate` (default)
- `--lane-by entry_gate`
- `--lane-by route` (entry -> exit)

Example:

```powershell
python visualize_trajectories.py --csv trajectories.csv --image path\to\drone_frame.png --lane-by route --output output\route_overlay.png
```

## Helpful Options

- `--max-tracks 200` to limit rendered trajectories for quick previews
- `--line-width 2.0` to adjust line thickness
- `--alpha 0.8` to adjust transparency
- `--legend-limit 20` to control legend size

## Notes

- The parser is designed for DataFromSky CSV rows where trajectory samples are stored as repeating groups of: `x, y, speed, acc, time`.
- If `--image` is not provided, trajectories are rendered on a generated canvas.
