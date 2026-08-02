# EuRoC Double-Adjoint Memory Benchmark

Objective for both methods: `1e6 * mean(Log(right-invariant error)^2)`.
Analytic outer gradients use `invariant_error=right`, `residual_jacobian=left`, `inner_vjp=local_ad`.
Batch size: `1000`, measured steps per case: `5`, warmup steps: `1`, device: `cuda`.

| W | Stage | AD time s | Adj time s | Time ratio adj/AD | AD max MB | Adj max MB | Memory ratio adj/AD | Status |
|---:|:---|---:|---:|---:|---:|---:|---:|:---|
| 16 | gyro | 0.1998 | 0.2149 | 1.076 | 396.5 | 337.8 | 0.852 | ok; ok |
| 16 | acc | 0.2199 | 0.2332 | 1.061 | 525.6 | 351.3 | 0.668 | ok; ok |
| 32 | gyro | 0.3639 | 0.3491 | 0.959 | 468.0 | 342.8 | 0.733 | ok; ok |
| 32 | acc | 0.3875 | 0.3699 | 0.955 | 745.5 | 366.9 | 0.492 | ok; ok |
| 64 | gyro | 0.7246 | 0.6857 | 0.946 | 494.0 | 234.7 | 0.475 | ok; ok |
| 64 | acc | 0.7741 | 0.7211 | 0.932 | 1069.3 | 279.8 | 0.262 | ok; ok |
| 128 | gyro | 1.5201 | 1.5199 | 1.000 | 898.5 | 370.0 | 0.412 | ok; ok |
| 128 | acc | 1.6132 | 1.6021 | 0.993 | 2067.2 | 457.0 | 0.221 | ok; ok |
