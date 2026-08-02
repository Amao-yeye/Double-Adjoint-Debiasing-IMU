# 跨主机复现说明

## 1. 已固定的研究环境

开发环境记录：Ubuntu 20.04.6、Python 3.10.12、PyTorch 2.4.0（原机 CUDA 12.1 build）、NumPy 2.0.1、SciPy 1.14.0。Python 依赖精确版本见 `requirements.txt`，Conda 入口见 `environment.yml`。

推荐使用 Ubuntu 20.04/22.04、Python 3.10 和 NVIDIA GPU。CPU 可运行 smoke test 和小规模检查，完整训练应使用 CUDA GPU。不同 GPU 型号、驱动和并行归约顺序可能导致末位浮点差异，因此复现判断应以保存的配置和指标容差为准，而不是要求 checkpoint 二进制逐字节一致。

## 2. 获取源码与建立环境

```bash
git clone --branch PLATO_formal --single-branch \
  git@github.com:Amao-yeye/Double-Adjoint-Debiasing-IMU.git
cd Double-Adjoint-Debiasing-IMU
bash scripts/create_env.sh
source .venv/bin/activate
```

也可使用 Conda：

```bash
conda env create -f environment.yml
conda activate plato-formal
```

PyTorch 版本固定为 2.4.0。需要指定 CUDA wheel 时，请保持 PyTorch 版本不变并按目标主机驱动选择官方 wheel index。

## 3. AquaLux 数据布局

将 `0524_data` 原样复制到新主机，然后选择以下一种方式：

```bash
# 推荐：数据位于独立磁盘
export AQUALUX_DATA_ROOT=/data/AquaLux/0524_data

# 或使用仓库默认路径
mkdir -p datasets/AquaLux
ln -s /data/AquaLux/0524_data datasets/AquaLux/0524_data
```

每个用于训练/测试的序列至少需要：

```text
$AQUALUX_DATA_ROOT/0524/<Sequence>/imu_diagnostics/imu_left.csv
$AQUALUX_DATA_ROOT/0524/<Sequence>/ATO/gt_imu_gravity_aligned.tum
$AQUALUX_DATA_ROOT/0524/<Sequence>/ATO/vo_imu_gravity_aligned.tum  # VO supervision
$AQUALUX_DATA_ROOT/<Sequence>.bag                                # OpenVINS
```

校验复制结果：

```bash
python tools/verify_aqualux_layout.py
```

校验器会检查默认正式训练/测试序列、必要文件和时间戳可解析性；IMU/GT 必须严格递增，VO 允许数据中已有的重复帧时间戳但禁止回跳。可用 `--require-vo`、`--require-bags` 强制检查对应阶段的输入。

## 4. 源码 smoke test

```bash
bash scripts/smoke_test.sh
```

该命令编译检查全部 Python 文件，并执行 PLATO loss、NLL 和 supervision smoke tests。若暂时没有数据，可运行：

```bash
bash scripts/smoke_test.sh --skip-data
```

## 5. AquaLux 缓存与正式训练

先生成/核验 native-rate 数据缓存：

```bash
python external/Debias_IMU/BiasDy/mainAquaLuxPLATO.py \
  --data_dir "$AQUALUX_DATA_ROOT" \
  --outputdir Results/AquaLux/preintegration/reproduction_gt_se23 \
  --cache_only --recompute_data
```

从头运行 GT-SE₂(3) MSE 的 gyro + accelerometer 训练和测试：

```bash
python external/Debias_IMU/BiasDy/mainAquaLuxPLATO.py \
  --data_dir "$AQUALUX_DATA_ROOT" \
  --data_cache_dir Results/AquaLux/preintegration/reproduction_gt_se23/cache \
  --outputdir Results/AquaLux/preintegration/reproduction_gt_se23 \
  --plato_supervision gt_se23 \
  --plato_stage mse \
  --gradient_mode double_adjoint \
  --plato_window_steps 64 \
  --loss_window 64 \
  --epoch_mse 1801 \
  --device cuda --test_device cuda \
  --run_test
```

训练代码固定使用种子 `3407`，数据层固定使用种子 `256`。每次运行会保存 `run_config.json`、timing report、checkpoint 和 metrics。

VO-NLL 的分阶段配方位于：

```bash
AQUALUX_DATA_ROOT="$AQUALUX_DATA_ROOT" \
PYTHON_BIN="$(command -v python)" \
bash tools/run_aqualux_vo_nll_alt300_bgba_cov.sh
```

该配方需要先生成脚本中标明的 MSE 初始化 checkpoint；也可以通过修改 `BG_INIT`、`BA_INIT` 指向本次 MSE 训练输出。

## 6. OpenVINS

OpenVINS 复现实验使用固定上游提交 `69488123ed9362dd44b6f28e7f4680abbff1442b`，并应用 `patches/external/openvins_local_changes_2026-05-18.patch`。在已安装 ROS Noetic 与 catkin-tools 的 Ubuntu 20.04 主机上运行：

```bash
bash scripts/setup_openvins.sh
export OPENVINS_ROOT="$PWD/external/OpenVINS"
```

之后可先 dry-run 检查命令与配置：

```bash
python tools/run_aqualux_openvins_raw.py \
  --data-root "$AQUALUX_DATA_ROOT" \
  --openvins-root "$OPENVINS_ROOT" \
  --sequences Test_009 --dry-run
```

去掉 `--dry-run` 运行正式评估。脚本会写出每个序列的固定标定配置、manifest、OpenVINS 轨迹和 evo 指标。

## 7. 输出核验

优先对照以下文件：

- `Results/reproducibility_snapshot/aqualux_preintegration_gt_se23_mse_metrics.csv`
- `Results/reproducibility_snapshot/aqualux_preintegration_vo_nll_final_metrics.csv`
- `Results/reproducibility_snapshot/aqualux_openvins_raw_metrics.csv`
- `Results/reproducibility_snapshot/aqualux_openvins_plato_vo_se3_mse_acc_nogate_metrics.csv`
- `Results/double_adjoint_memory_benchmark/summary.csv`

运行产物、缓存、ROS bag 和 checkpoint 默认被 `.gitignore` 排除。它们属于可再生成数据，不是源码缺失。

## 8. 故障定位

- 找不到数据：确认 `AQUALUX_DATA_ROOT` 直接指向含 `0524/` 和 `*.bag` 的 `0524_data`。
- CUDA 不可用：先用 CPU 跑 smoke test，再检查 NVIDIA driver 与 PyTorch CUDA build。
- OpenVINS 构建失败：确认系统为 ROS Noetic 支持环境，且 `/opt/ros/noetic/setup.bash` 存在。
- evo 导入失败：确认当前激活的是本项目环境；不再依赖任何 `/home/amao/.local/...` 路径。
