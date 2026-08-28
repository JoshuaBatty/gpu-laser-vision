<div align="center">

# GPU Laser Vision

**Real-time Rust GPU vision that turns live video into segmented projector output and coloured laser paths.**

[![Rust 2024](https://img.shields.io/badge/Rust-2024-B7410E?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Nix](https://img.shields.io/badge/dev_shell-Nix-5277C3?logo=nixos&logoColor=white)](https://nixos.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-663399)](LICENSE)

</div>

[![GPU Laser Vision dashboard — click to watch the 1080p demo](docs/media/dashboard.png)](https://github.com/user-attachments/assets/a6500e34-b6f1-441e-bf42-8834c9743e6e)

GPU Laser Vision compares two live 720p pipelines side by side: a custom CUDA
edge detector and CUDA-backed YOLO11 instance segmentation. Both recover source
colours, extract contours, and produce normalized paths compatible with
`nannou_laser` while the interface exposes selected intermediate stages and
aggregate callback costs. The result can be sent to a fullscreen secondary
display or streamed to an Ether Dream DAC with live CUDA/YOLO source selection.

## Live demo

https://github.com/user-attachments/assets/a6500e34-b6f1-441e-bf42-8834c9743e6e

## At a glance

| | Vision baseline (2026-08-27) |
|---|---:|
| Input | 1280 × 720 RGBA video at 25 FPS |
| Delivered rate | **24.8–25.2 FPS** |
| Processing callback | **14.8–15.6 ms** |
| YOLO11n segmentation | 9.1–10.1 ms |
| Custom CUDA edge pipeline | 1.8–1.9 ms |
| Display texture updates | 1.9 ms |
| Test GPU | NVIDIA GeForce RTX 5090 |

The media is the limiting clock: this measured callback had approximately
64–68 FPS of compute capacity, but a 25 FPS file cannot deliver additional
unique frames. These figures predate the projector and physical-output branch
and remain the last recorded vision-pipeline baseline. See
[the profiling notes](docs/profiling.md) for methodology, stage breakdowns, and
the Nsight Scharr experiment.

## Architecture

```mermaid
flowchart LR
    Video[720p RGBA video]

    Video --> CUDA[Custom CUDA graph]
    CUDA --> Gray[Grayscale]
    Gray --> Scharr[Scharr magnitude]
    Scharr --> CMask[Threshold + colour recovery]
    CMask --> CPath[Coloured CUDA path]

    Video --> YOLO[YOLO11n-seg on CUDA]
    YOLO --> PMask[Person mask]
    PMask --> Contour[Contour + colour recovery]
    Contour --> YPath[Coloured YOLO path]

    CPath --> UI[Live nannou / egui dashboard]
    YPath --> UI
    Video --> Isolate[Person isolation]
    PMask --> Isolate
    Isolate --> Projector[Fullscreen projector output]
    CPath --> Pack[Scanner-aware frame packing]
    YPath --> Pack
    Pack --> DAC[Ether Dream DAC]
```

### Custom CUDA path

1. Convert packed RGBA pixels to normalized luminance.
2. Compute Scharr magnitude and horizontal/vertical gradients.
3. Apply the live threshold range and sample a strong source colour across the
   gradient normal.
4. Copy compact display outputs into pinned host memory.
5. Trace active mask pixels into open paths, closed loops, and isolated points.

The kernels run as a captured CUDA graph with persistent device buffers.
Threshold values are uploaded only when the controls change.

### Neural path

1. Upload, resize, normalize, and letterbox the frame on CUDA.
2. Run the fixed 640 × 640 YOLO11n-seg TorchScript model with LibTorch.
3. Decode the strongest COCO `person` instance and restore its mask to 720p.
4. Extract a one-pixel contour and replace green spill with nearby foreground
   colour.
5. Feed the same path tracer used by the classical pipeline.

This keeps the comparison honest: the two pipelines differ in perception, then
share the same colour, geometry, display, and point-count surfaces.

### Physical outputs

- The projector view isolates the detected person against black and opens as a
  dedicated borderless window on the secondary display. The main dashboard
  remains unchanged.
- The Ether Dream worker continuously maintains the DAC buffer at up to 30,000
  points per second and swaps geometry only at complete frame boundaries.
- YOLO contours and dense CUDA edges use separate scanner-packing policies so
  coherent silhouettes stay smooth while fragmented edges are joined, ordered,
  resampled, and bounded to a practical physical frame.
- Laser output is disabled by default and must be enabled explicitly from the
  interface.

## Run it

### Requirements

- Linux or WSL2 with an NVIDIA GPU and working host driver
- [Nix](https://nixos.org/download/) with flakes enabled
- Git and internet access for the first dependency/model fetch
- A local video at `assets/jcvd_green_screen_720p.mp4`

CUDA, nightly Rust, LibTorch/PyTorch 2.11, FFmpeg, `uv`, and the native windowing
libraries are pinned by `flake.nix`. No system CUDA toolkit is required.

```bash
git clone https://github.com/JoshuaBatty/gpu-laser-vision.git
cd gpu-laser-vision

# Download YOLO11n-seg and export the fixed TorchScript artifact.
nix develop --command uv run scripts/export_yolo.py

# Supply your own 1280 × 720 demo clip at the path expected by the app.
cp /path/to/your/video.mp4 assets/jcvd_green_screen_720p.mp4

# Build the CUDA kernels and launch the release application.
nix develop --command cargo oxide run
```

The first build compiles the CUDA codegen backend and LibTorch C++ bridge, so it
is substantially slower than subsequent launches.

The source clip is intentionally gitignored to keep the repository small; use a
720p green-screen clip to reproduce the presentation shown above. Generated
`.pt` and `.torchscript` model files are also ignored and can always be rebuilt
by the export script.

A second display and Ether Dream DAC are optional. Projector output is offered
only when a secondary monitor is detected. Ether Dream uses network discovery,
then falls back to `ETHER_DREAM_IP` (default `192.168.0.2`) for environments
such as WSL where LAN broadcasts may not arrive:

```bash
ETHER_DREAM_IP=192.168.0.2 nix develop --command cargo oxide run
```

## Interface

- **Edge thresholds** adjust the inclusive normalized Scharr range in real time.
- **CUDA / YOLO cards** compare coloured contours and generated paths.
- **Diagnostics** expose grayscale, Scharr, edge-mask, and person-mask stages.
- **Performance** reports rolling FPS and per-stage callback timings.
- **Projector output** opens the isolated person feed on the secondary display.
- **Laser controls** select CUDA or YOLO geometry and explicitly enable output.
- **Status chips** show CUDA graph, YOLO confidence, and Ether Dream connection.

All previews retain their 16:9 source geometry and update on every decoded
frame. The dashboard has been verified live at 1280 × 720 and 1920 × 1080.

## Repository map

| Path | Responsibility |
|---|---|
| [`src/edge_detection.rs`](src/edge_detection.rs) | Persistent CUDA graph, buffers, and host reconstruction |
| [`src/kernels.rs`](src/kernels.rs) | Grayscale, Scharr, threshold, and colour-recovery kernels |
| [`src/yolo.rs`](src/yolo.rs) | TorchScript inference, mask decoding, and contour colourization |
| [`src/path_generation.rs`](src/path_generation.rs) | Mask graph traversal and normalized laser geometry |
| [`src/laser.rs`](src/laser.rs) | Ether Dream discovery, scanner-frame packing, and background streaming |
| [`src/interface.rs`](src/interface.rs) | Responsive dashboard layout and visual primitives |
| [`docs/profiling.md`](docs/profiling.md) | End-to-end and isolated-kernel measurements |
| [`scripts/export_yolo.py`](scripts/export_yolo.py) | Reproducible Ultralytics → TorchScript export |

## Current scope

The complete source → GPU vision → projector / Ether Dream path is implemented.
The next phase is production installation work: live-camera capture, projector
and scanner calibration, venue-specific safety integration, and recording the
finished light performance.

Laser hardware requires appropriate scan limits, blanking, interlocks, and
venue-specific safety controls. The on-screen paths alone should not be treated
as a complete laser-safety system.

## License and attribution

This project is released under the [GNU Affero General Public License v3.0](LICENSE).

The model export workflow uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
and the generated YOLO11 model is subject to Ultralytics' AGPL-3.0 terms unless
covered by a separate enterprise license. Model weights are not committed.
