# Profiling

Baseline: `scharr` over 640,000 pixels on an RTX 5090 (CC 12.0), driver
576.88. Nsight Systems 2025.6.3; Nsight Compute 2025.2.1.

```bash
nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none --stats=true --output=target/nsys-scharr ./target/release/nannou-laser-cuda
ncu --set full --kernel-name regex:scharr --launch-count 1 --export target/ncu-scharr ./target/release/nannou-laser-cuda
```

| Metric | Baseline |
|---|---:|
| Scharr (Systems) | 3.96 us |
| All kernels | 326 us |
| 64 hysteresis launches | 313 us |
| Device-to-host copies | 1.54 ms |
| Scharr (Compute) | 6.82 us |
| DRAM / L2 throughput | 21.5% / 55.9% |
| L1 / L2 hit rate | 51.9% / 39.4% |
| Long-scoreboard stalls | 36.0% |
| Achieved occupancy | 78.5% |
| Excessive global-load sectors | 12.0% |

Shared-memory variants took 4.74-6.30 us, so the original kernel remains in use.
Timings from Systems and Compute are not directly comparable.
