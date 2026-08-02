# PLATO Formal：Double-Adjoint IMU Debiasing

本分支是 PLATO / Double-Adjoint IMU bias dynamics 的可迁移研究快照，包含当前源码、AquaLux 适配、PLATO 的 SE₂(3) / NLL / double-adjoint 实现、实验脚本、OpenVINS 补丁、关键结果摘要和研究记录。

大型 AquaLux 数据、训练 checkpoint、缓存和完整 OpenVINS 输出不进入 Git；它们均可由本仓库脚本重新生成。AquaLux 数据由使用者复制到新主机。

## 最短复现路径

```bash
git clone --branch PLATO_formal --single-branch \
  git@github.com:Amao-yeye/Double-Adjoint-Debiasing-IMU.git
cd Double-Adjoint-Debiasing-IMU

bash scripts/create_env.sh
source .venv/bin/activate

# 将 AquaLux 的 0524_data 放到任意位置，并设置：
export AQUALUX_DATA_ROOT=/absolute/path/to/AquaLux/0524_data
python tools/verify_aqualux_layout.py
bash scripts/smoke_test.sh
```

若数据放在仓库默认位置，则无需设置环境变量：

```text
datasets/AquaLux/0524_data/
├── 0524/Train_000/...
├── 0524/Test_001/...
├── Train_000.bag
└── Test_001.bag
```

完整的环境、数据、训练、OpenVINS 和结果核验流程见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

## 目录

```text
external/Debias_IMU/       上游 Debias_IMU 与本项目当前修改的完整源码快照
external/Debias_IMU/BiasDy/plato/
                           PLATO SE₂(3)、NLL、监督、协方差和 double-adjoint 实现
tools/                     AquaLux 训练、导出、OpenVINS 和绘图工具
patches/external/          固定版本外部依赖补丁
docs/                      方法、实验和推导记录
Results/                   小型汇总表、关键指标快照和内存 benchmark
papers/                    研究参考资料
```

## 主要入口

- `external/Debias_IMU/BiasDy/mainAquaLuxPLATO.py`：AquaLux PLATO 训练/评估主入口。
- `external/Debias_IMU/BiasDy/mainEurocPLATO.py`：EuRoC PLATO 主入口。
- `tools/run_aqualux_raw_preintegration.py`：AquaLux raw IMU 预积分基线。
- `tools/run_aqualux_openvins_raw.py`：AquaLux OpenVINS 运行与评估。
- `tools/run_aqualux_vo_nll_alt300_bgba_cov.sh`：VO-NLL 分阶段训练配方。
- `Results/reproducibility_snapshot/`：用于对照复现输出的关键指标与配置。

## 来源与许可

Debias_IMU 基线来自 UMich-CURLY/Debias_IMU，其许可证保留在 `external/Debias_IMU/LICENSE`。外部论文和第三方代码保留各自的作者、引用与许可要求。

