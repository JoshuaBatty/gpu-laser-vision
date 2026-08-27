# Profiling

## End-to-end live vision pipeline

Measured on 2026-08-27 with the release build and the complete UI running.
The first inference and its CUDA/cuDNN warm-up are excluded; subsequent values
are ranges from two-second averaging windows printed by the application.

| Configuration | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (CC 12.0) |
| Driver | 610.88 |
| Input | 1280x720 RGBA video at 25 FPS |
| Model | YOLO11n-seg TorchScript, fixed 640x640 input |
| Display policy | Every processed panel updated on every source frame |

```bash
nix develop --command cargo oxide run
```

| Metric | Baseline | Optimized |
|---|---:|---:|
| Displayed source frames | 22-24 FPS | 24.8-25.2 FPS |
| Total processing callback | 30.8-32.1 ms | 14.8-15.6 ms |
| YOLO segmentation | 15.6-16.4 ms | 9.1-10.1 ms |
| CUDA edge detection | 1.8-2.1 ms | 1.8-1.9 ms |
| CUDA laser-path tracing | 2.9-4.4 ms | 1.2-1.8 ms |
| YOLO laser-path tracing | 0.8-1.0 ms | 0.1-0.2 ms |
| Video-frame copy | 0.3 ms | 0.3 ms |
| Display-texture updates | 8.1-8.7 ms | 1.9 ms |
| Callback capacity (`1000 / total_ms`) | 31-32 FPS | 64-68 FPS |

The baseline texture value is averaged across all source frames even though
those textures were rebuilt and uploaded only once every three frames. The
optimized pipeline reuses their allocations and updates them on every frame.

The displayed rate is capped by the media, not the vision pipeline:
`jcvd_green_screen_720p.mp4` contains exactly 25 frames per second. Callback
capacity is the inverse of measured processing latency, not a claim that this
25 FPS file can produce additional unique frames. A higher-rate camera or test
asset is required to measure delivered throughput above 25 FPS.

The optimized path:

- performs YOLO resize, RGB conversion, normalization, and letterboxing on CUDA;
- reuses the fixed-size YOLO input tensor and enables cuDNN benchmarking;
- gathers detection metadata with one device-to-host synchronization;
- thresholds the restored mask to `u8` before copying it to the host;
- restricts contour work to the detected bounding box;
- carries sorted active-pixel indices into laser tracing instead of repeatedly
  scanning the complete 921,600-pixel frame;
- uses compact per-pixel direction bits instead of hashing every traced edge;
- reuses Bevy texture allocations while updating every debug texture every
  frame; and
- avoids unchanged threshold uploads and an unnecessary full-frame clone.

An FP16 TorchScript model/input experiment did not improve steady-state YOLO
latency and introduced mixed output types requiring an additional cast, so it
was reverted. The TorchScript network remains the dominant stage; TensorRT is
the likely next step for a material inference improvement.

## Isolated Scharr kernel

Historical baseline: `scharr` over 640,000 pixels on an RTX 5090 (CC 12.0),
driver 576.88. Nsight Systems 2025.6.3; Nsight Compute 2025.2.1.

```bash
nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none --stats=true --output=target/nsys-scharr ./target/release/nannou-laser-cuda
ncu --set full --kernel-name regex:scharr --launch-count 1 --export target/ncu-scharr ./target/release/nannou-laser-cuda
```

| Metric | Baseline |
|---|---:|
| Scharr (Systems) | 3.96 us |
| Scharr (Compute) | 6.82 us |
| DRAM / L2 throughput | 21.5% / 55.9% |
| L1 / L2 hit rate | 51.9% / 39.4% |
| Long-scoreboard stalls | 36.0% |
| Achieved occupancy | 78.5% |
| Excessive global-load sectors | 12.0% |

Shared-memory variants took 4.74-6.30 us, so the original kernel remains in use.
Timings from Systems and Compute are not directly comparable.
