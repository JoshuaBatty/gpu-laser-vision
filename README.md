# gpu-laser-vision
Rust GPU vision pipeline turning images and live video into laser geometry. Custom CUDA kernels, Nsight-profiled performance experiments, and PyTorch segmentation feed contour extraction and laser-safe paths for real-time output.

Generate the gitignored YOLO TorchScript artifact before the first run:

```sh
nix develop --command uv run scripts/export_yolo.py
```

Use the left-side **Edge thresholds** panel to select the inclusive normalized
Scharr-magnitude range sent to the laser-edge kernel.

Live processing uses `assets/jcvd_green_screen_720p.mp4` when that local,
gitignored clip is present, and otherwise falls back to the tracked Big Buck
Bunny sample.
