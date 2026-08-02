
---

## 给 Codex 的实现指示

你在一个 fresh clone 的 `Debias_IMU-main` 上修改，用于实现 PLATO 在 EuRoC dataset 上的第一版验证。目标是先完成两种监督：

1. **高频 GT 监督**：使用 EuRoC ground truth 的 `R, v, p`，监督 SE2(3) full state。
2. **低频但覆盖更长时间的 VO 监督**：使用较低频率 VO/VIO pose 作为监督，只监督 `R, p`，窗口覆盖时间要比高频 GT 更长。

### 0. 硬约束

不要改变原始网络结构。不要改 `bw_func_net`、`ba_func_net`、`VFSE23vbiasBase` 的网络层、输入输出维度或 forward dynamics。允许新增 loss、covariance/NLL 模块、dataset sampling helper、训练入口脚本和 CLI 参数。核心修改必须是 loss function 和 supervision sampling，不是换网络。

保留原始训练策略：MSE 阶段仍然先训练 gyro，再 freeze gyro 训练 acc。NLL 阶段从 MSE 训练好的 gyro+acc 权重开始 finetune。

当前仓库关键位置：

* `BiasDy/learning.py`

  * `Gyro_train()` 当前只用 `SO3log_loss`
  * `Acc_train()` 当前直接对 `v,p` 做 MSE
* `BiasDy/BiasDy_loss.py`

  * 当前只有 `SO3log_loss`
* `BiasDy/lie_algebra.py`

  * 已有 `SEn3log()`、`SEn3inverse()`、`SEn3fromSO3Rn()`、`SO3leftJacoInv()`，要优先复用
* `BiasDy/dataset.py`

  * `get_data_for_biasDy()` 当前返回 dense GT windows
* `BiasDy/VIO_visualization.py`

  * 已有 VO/VIO text 文件读取、pose interpolation、alignment helper，可复用或迁移

新代码可以新增 `BiasDy/plato_loss.py`，或者扩展 `BiasDy/BiasDy_loss.py`。建议新增 `BiasDy/plato_loss.py`，保持原始 baseline loss 文件尽量干净。

---

## 1. 实现 SE2(3) residual 和第一阶段 MSE loss

### 1.1 Full SE2(3) residual

实现：

```python
r_k = log(X_gt,k^{-1} X_pred,k)
```

这里：

```text
X = [ R  v  p
      0  I  0 ]
```

更具体地：

```text
X_err = X_ref^{-1} X_pred

R_error = R_ref^T R_pred
v_error = R_ref^T (v_pred - v_ref)
p_error = R_ref^T (p_pred - p_ref)

phi = SO3log(R_error)
[rho_v, rho_p] = J_l(phi)^(-1) [v_error, p_error]

r = [phi, rho_v, rho_p]  # shape (..., 9)
```

不要使用当前旧的 `SO3log(R_pred^T R_gt)` convention。PLATO 新 loss 全部使用用户要求的 `X_ref^{-1} X_pred`。

可以直接复用 `Lie.SEn3inverse()` 和 `Lie.SEn3log()`：

```python
def make_se23_from_solution(R_pred, solution):
    # solution order in current code: [xi, t, bw, v, p, ba]
    v_pred = solution[..., 7:10]
    p_pred = solution[..., 10:13]
    return Lie.SEn3fromSO3Rn(R_pred, torch.cat([v_pred, p_pred], dim=-1))

def se23_log_residual(X_ref, R_pred, solution):
    X_pred = make_se23_from_solution(R_pred, solution)
    X_err = Lie.SEn3inverse(X_ref) @ X_pred
    return Lie.SEn3log(X_err)  # [..., 9], order [phi, rho_v, rho_p]
```

确认 `Lie.SEn3log()` 内部确实是：

```python
phi = SO3log(X[..., :3, :3])
temp_rest = SO3leftJacoInv(phi) @ X[..., :3, 3:]
```

这正好满足 `J_l(phi)^(-1) * [v_error, p_error]`。

### 1.2 Full state MSE loss

实现：

```python
L_mse_se23 = 0.5 * mean(||W_r * r||^2)
```

支持 block weights：

```text
rot_weight
vel_weight
pos_weight
```

默认建议：

```text
rot_weight = 1.0
vel_weight = 1.0
pos_weight = 1.0
```

但训练阶段使用不同 blocks：

* Gyro MSE 阶段：只用 rotation residual，即 `[phi]`
* Acc MSE 阶段：gyro 已 frozen，主要用 velocity 和 position residual，即 `[rho_v, rho_p]`
* full-state logging 可以记录完整 `[phi, rho_v, rho_p]`

### 1.3 VO pose residual

VO 监督只有 pose，没有 velocity。实现 pose residual：

```text
r_pose = log(T_vo^{-1} T_pred)
       = [phi, rho_p]  # shape (..., 6)
```

不要把 VO pose 硬塞成 full SE2(3) velocity target。正确做法是只监督 `R,p`：

```python
def se23_pose_residual(R_ref, p_ref, R_pred, p_pred):
    R_err = R_ref.transpose(-1, -2) @ R_pred
    phi = Lie.SO3log(R_err)
    p_err = torch.matmul(
        R_ref.transpose(-1, -2),
        (p_pred - p_ref).unsqueeze(-1)
    ).squeeze(-1)
    rho_p = torch.matmul(
        Lie.SO3leftJacoInv(phi),
        p_err.unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat([phi, rho_p], dim=-1)
```

VO MSE 阶段：

* Gyro：只用 `phi`
* Acc：用 `rho_p`
* 不使用 velocity residual，因为 VO 没有 velocity

---

## 2. 实现 covariance 的离散化解

新增 covariance helper。不要形成 dense marginal covariance `P_s`。可以构造 block-tridiagonal precision `Λ_s`，必要时为了第一版实现可以用 dense tensor 存这个 precision matrix，但不要通过 dense covariance inverse 得到它。

### 2.1 SE2(3) invariant error dynamics

状态误差顺序固定为：

```text
xi = [phi, rho_v, rho_p]  # dim = 9
```

使用：

```text
dot(xi) = F xi + G(t) n
n = [n_a, n_g]
```

其中：

```text
F = [ 0      0   0
      g^     0   0
      0      I   0 ]
```

`g = [0, 0, -9.81]`，与当前 `VFSE23vbiasBase.g_const` 保持一致。

解析 transition：

```text
Φ(Δt) =
[ I              0     0
  g^ Δt          I     0
  0.5 g^ Δt^2    Δt I  I ]
```

实现：

```python
def se23_phi(dt, device, dtype):
    # returns (..., 9, 9)
```

### 2.2 每个 IMU step 的离散噪声注入矩阵

按照论文里的离散形式实现：

```text
G_d,k = -Δt_k *
[ 0              R_k
  R_k            v_k^ R_k
  0              p_k^ R_k ]
```

shape 是 `9 x 6`，列顺序是 `[n_a, n_g]`，行顺序是 `[phi, rho_v, rho_p]`。

实现：

```python
def se23_Gd(R, v, p, dt):
    # R: (..., 3, 3)
    # v: (..., 3)
    # p: (..., 3)
    # dt: scalar or (...,)
    # return (..., 9, 6)
```

如果采用 continuous noise density convention，则使用不带 `Δt` 的 `G_c` 并令：

```text
Q_d,k = G_c,k Q_c G_c,k^T Δt_k
```

但为了和上面的离散公式一致，第一版采用 per-sample convention：

```text
Q_d,k = G_d,k Q_n G_d,k^T
Q_n = diag(σ_a^2, σ_a^2, σ_a^2, σ_g^2, σ_g^2, σ_g^2)
```

在代码注释中明确这个 convention。

### 2.3 一个监督区间内的 preintegrated covariance

监督时间 `t_i^s` 到 `t_{i+1}^s` 之间有多个 IMU step。离散 covariance 解：

```text
Q_i^s =
Σ_{j=s_i}^{s_{i+1}-1}
Φ(t_{i+1}^s - t_{j+1}) Q_d,j Φ(t_{i+1}^s - t_{j+1})^T
```

也可以递推实现：

```python
P = 0
Phi_total = I
for j in range(start_idx, end_idx):
    Phi_j = se23_phi(dt_j)
    Gd_j = se23_Gd(R_j, v_j, p_j, dt_j)
    Qd_j = Gd_j @ Qn @ Gd_j.transpose(-1, -2)
    P = Phi_j @ P @ Phi_j.transpose(-1, -2) + Qd_j
    Phi_total = Phi_j @ Phi_total

Q_s_i = P
Phi_s_i = Phi_total
```

由于 `F` constant，也可以令：

```text
Φ_i^s = Φ(t_{i+1}^s - t_i^s)
```

但递推版更容易支持非均匀 timestamps。

所有 covariance matrix 都要：

```python
Q = 0.5 * (Q + Q.transpose(-1, -2))
Q = Q + jitter * I
```

默认 `jitter = 1e-8` 或 `1e-6`，用 CLI 暴露。

### 2.4 构造 block-tridiagonal precision

监督状态数为 `M+1`，每个状态维度 `dx=9`。给定：

```text
xi_{i+1} = Phi_i^s xi_i + nu_i^s
nu_i^s ~ N(0, Q_i^s)
```

初始 prior：

```text
xi_0 ~ N(0, P_0^s)
```

由于训练窗口 initial condition 来自监督轨迹，`P_0^s` 设成小正定，例如：

```text
P_0^s = prior_std^2 I
prior_std = 1e-4 or 1e-3
```

precision 非零 blocks：

```text
Λ_00 = P0^{-1} + Phi_0^T Q_0^{-1} Phi_0

Λ_ii = Q_{i-1}^{-1} + Phi_i^T Q_i^{-1} Phi_i,
       i = 1, ..., M-1

Λ_MM = Q_{M-1}^{-1}

Λ_i,i+1 = - Phi_i^T Q_i^{-1}
Λ_i+1,i = - Q_i^{-1} Phi_i
```

实现函数：

```python
def build_se23_precision(Phi_list, Q_list, prior_std, jitter):
    # Phi_list length M, each [B, 9, 9]
    # Q_list length M, each [B, 9, 9]
    # return Lambda [B, (M+1)*9, (M+1)*9]
```

第一版可以用 dense storage 表示 `Λ` 和 `K`，但必须来自 block-tridiagonal assembly，不要先形成 dense covariance 再 inverse。

---

## 3. 实现第二阶段 NLL loss

### 3.1 通用 NLL 公式

监督 residual stack：

```text
r = [r_0, ..., r_M]
```

measurement model：

```text
r = H xi_s + e
e ~ N(0, W)
```

用 Woodbury 形式：

```text
K = Λ_s + H^T W^{-1} H
η = H^T W^{-1} r
```

NLL：

```text
L =
0.5 r^T W^{-1} r
- 0.5 η^T K^{-1} η
+ 0.5 logdet(W)
+ 0.5 logdet(K)
- 0.5 logdet(Λ_s)
```

实现时用 Cholesky：

```python
chol_K = torch.linalg.cholesky(K + jitter * I)
solve_K_eta = torch.cholesky_solve(eta.unsqueeze(-1), chol_K).squeeze(-1)
logdet_K = 2 * torch.diagonal(chol_K, dim1=-2, dim2=-1).log().sum(-1)
```

同理计算 `logdet_Λ`。如果 Cholesky 失败，先 symmetrize，再增大 jitter。

### 3.2 Full SE2(3) GT NLL

GT full-state supervision：

```text
r_i ∈ R^9
H_i = I_9
W_i = diag(σ_R_gt^2 I3, σ_v_gt^2 I3, σ_p_gt^2 I3)
```

支持 CLI：

```text
--meas_sigma_rot_gt
--meas_sigma_vel_gt
--meas_sigma_pos_gt
```

默认可以先用：

```text
σ_R_gt = 1e-3 rad
σ_v_gt = 1e-2 m/s
σ_p_gt = 1e-3 or 1e-2 m
```

### 3.3 VO pose NLL

VO pose supervision：

```text
r_i ∈ R^6 = [phi, rho_p]
H_i =
[ I3  0   0
  0   0   I3 ]
```

measurement covariance：

```text
W_i = diag(σ_R_vo^2 I3, σ_p_vo^2 I3)
```

支持 CLI：

```text
--meas_sigma_rot_vo
--meas_sigma_pos_vo
```

默认可以先用：

```text
σ_R_vo = 1e-2 rad
σ_p_vo = 5e-2 m
```

### 3.4 Learnable IMU noise covariance

NLL 阶段需要优化 IMU noise covariance。不要把它塞进网络结构。新增一个独立 `torch.nn.Module` 或直接在 trainer 里放 `torch.nn.Parameter`：

```python
class PlatoNoiseParams(torch.nn.Module):
    def __init__(self, init_sigma_a=0.1, init_sigma_g=0.01, isotropic=True):
        self.log_sigma_a = nn.Parameter(...)
        self.log_sigma_g = nn.Parameter(...)

    def Qn(self):
        # diag [a,a,a,g,g,g]
```

支持 isotropic 和 per-axis：

```text
--plato_noise_isotropic true/false
```

NLL optimizer 包括：

```python
list(model.parameters()) + list(noise_params.parameters())
```

但不要修改 `VFSE23vbiasBase` 内部结构。

---

## 4. EuRoC supervision sampling

### 4.1 高频 GT supervision

可以基于当前 `dataset.get_data_for_biasDy()`，因为它已经返回：

```text
X_gt_batch: [T, B, 5, 5]
t_odeint:   [T]
solution:   [T, B, 16]
R_sol:      [T, B, 3, 3]
```

高频 GT 设置：

```text
supervised_indices = torch.arange(T)
```

MSE：

```python
r = se23_log_residual(X_gt_batch, R_sol, solution)
```

NLL：

* 对每个 supervised interval 构造 `Phi_i^s, Q_i^s`
* full-state `H=I9`
* 用 full-state NLL

### 4.2 低频长窗口 VO supervision

新增 sampling 逻辑，不要只用当前短 `loss_window`。需要区分：

```text
integration window length: window_imu_steps
supervision stride/rate:  sup_stride or sup_rate_hz
```

新增 CLI：

```text
--plato_window_steps
--plato_sup_stride
--plato_sup_rate_hz
```

对于 EuRoC `dt=0.005`：

```text
200 Hz IMU
5 Hz supervision => sup_stride = 40
10 Hz supervision => sup_stride = 20
```

VO 模式建议默认：

```text
plato_window_steps = 400  # 2 seconds at 200 Hz
plato_sup_stride = 40     # 5 Hz supervision
batch_size much smaller, e.g. 32 or 64
```

实现 VO loader：

* 支持 OpenVINS/VIO text 格式：

```text
timestamp px py pz qx qy qz qw
```

可复用 `BiasDy/VIO_visualization.py` 的 `read_vio_est_file()`。
新增 CLI：

```text
--vo_dir
--vo_template
```

例如：

```text
--vo_template "{seq}/00_estimate.txt"
```

或：

```text
--vo_template "{seq}.txt"
```

VO 对齐：

第一版为了 EuRoC 验证，可以支持：

```text
--vo_align first_gt
```

逻辑：

1. 找到 VO 和当前 sequence GT 的第一个共同/最近 timestamp。
2. 计算 `T_align = T_gt0 @ inv(T_vo0)`。
3. 所有 VO pose 变换到当前 EuRoC/GT frame。
4. 后续 loss 只用 VO pose，不用 GT pose。

也要支持：

```text
--vo_align none
```

方便之后使用已经对齐好的 VO。

VO interpolation：

* 对每个训练 window 的 supervised timestamps，从 VO pose 插值得到 `R_vo, p_vo`
* 旋转用 SO3 interpolation
* position 用 linear interpolation
* 如果 window 内有效 VO 点不足 2 个，resample start index

VO MSE：

```python
r_pose = se23_pose_residual(R_vo, p_vo, R_sol[sup_idx], p_pred[sup_idx])
```

VO NLL：

* 只对 `sup_idx` 构造 downsampled covariance intervals
* `H` 用 pose selection matrix
* residual 维度 6

---

## 5. 训练入口和训练策略

建议新增脚本：

```text
BiasDy/mainEurocPLATO.py
```

不要破坏原 `mainEuroc.py` baseline。

### 5.1 CLI 参数

至少加入：

```text
--plato_supervision {gt_se23,vo_pose}
--plato_stage {mse,nll,both}
--plato_window_steps
--plato_sup_stride
--batch_size
--epoch_mse
--epoch_nll
--lr_Gyro
--lr_Acc
--lr_NLL
--weight_decay_Gyro
--weight_decay_Acc
--weight_decay_NLL
--resume_gyro
--resume_acc
--vo_dir
--vo_template
--vo_align {first_gt,none}
--meas_sigma_rot_gt
--meas_sigma_vel_gt
--meas_sigma_pos_gt
--meas_sigma_rot_vo
--meas_sigma_pos_vo
--init_sigma_a
--init_sigma_g
--cov_jitter
--prior_std
```

### 5.2 MSE stage

保持原始 schedule：

#### Gyro MSE

```python
model.set_bw_zero(False)
model.set_ba_zero(True)
model.freeze_func_ba()
```

loss：

* GT mode：`SE23 residual` 的 rotation block `phi`
* VO mode：`pose residual` 的 rotation block `phi`

保存：

```text
outputdir/Gyro_weights/
```

#### Acc MSE

加载 gyro：

```python
bw_func.load_state_dict(...)
```

然后：

```python
model.set_bw_zero(False)
model.set_ba_zero(False)
model.freeze_func_bw()
```

loss：

* GT mode：`SE23 residual` 的 velocity+position blocks `[rho_v, rho_p]`
* VO mode：`pose residual` 的 position block `[rho_p]`

保存：

```text
outputdir/Acc_weights/
```

保留原始 `Test()` evaluation 路径。

### 5.3 NLL stage

从 MSE 权重开始：

```python
load gyro checkpoint
load acc checkpoint
model.set_bw_zero(False)
model.set_ba_zero(False)
```

默认建议 unfreeze both gyro and acc 做 finetune，但加 CLI：

```text
--nll_freeze_gyro
--nll_freeze_acc
```

optimizer：

```python
optimizer = Adam(
    trainable model params + noise_params params,
    lr=lr_NLL,
    weight_decay=weight_decay_NLL
)
```

NLL loss：

* `gt_se23`：full-state NLL
* `vo_pose`：pose NLL

保存：

```text
outputdir/NLL_weights/
```

同时保存 learned covariance：

```python
{
  "func_bw_model": ...,
  "func_ba_model": ...,
  "noise_params": noise_params.state_dict(),
  "optimizer": ...,
}
```

---

## 6. 需要修正/注意的现有小问题

`BiasDy/learning.py` 的 `Acc_train()` validation loop 里当前疑似用了 `dataset_train.get_data_for_biasDy()`，应该改成 `dataset_val.get_data_for_biasDy()`。这是 bug fix，不是网络结构变化。

`Lie.SEn3fromSO3Rn()` 里创建 identity 时最好补 `.to(R.dtype)`，避免 mixed dtype 问题。这是数值稳定修正，不改变网络。

所有新增 loss 都必须支持 shape：

```text
[T, B, ...]
[B, T, ...]
```

但训练里请统一成 `[T, B, ...]`，和当前 `odeint_SO3()` 返回一致。

不要在 loss 内部用 `.detach()`、`.item()` 或 numpy。Covariance/NLL 必须保持 torch differentiable。

---

## 7. 最小测试要求

新增 smoke/unit tests 或至少在脚本中提供 `--debug_smoke`。

必须检查：

1. **SE23 residual zero test**

```python
r = se23_log_residual(X_gt, R_gt, solution_gt_like)
assert max(abs(r)) < 1e-5
```

2. **small perturbation test**

给 `X_pred = X_gt exp(delta)`，确认 residual 近似等于 `delta`。

3. **pose residual zero test**

VO pose 与 pred pose 相同，`r_pose` 近似 0。

4. **covariance SPD test**

`Q_s_i`、`Λ_s`、`K` Cholesky 成功，NLL finite。

5. **gradient test**

跑一个 batch：

```python
loss.backward()
```

确认 gyro/acc 参数和 noise params 在对应 stage 有 non-None gradients。

6. **network shape unchanged**

保存修改前后 `bw_func_net`、`ba_func_net` parameter names/shapes，不允许新增或改变网络参数 shape。

---

## 8. 推荐运行命令

### 高频 GT：MSE

```bash
python3 BiasDy/mainEurocPLATO.py \
  --plato_supervision gt_se23 \
  --plato_stage mse \
  --outputdir results/Euroc_PLATO_gt_mse_w64 \
  --loss_window 64 \
  --plato_window_steps 64 \
  --batch_size 512 \
  --epoch_mse 1801
```

### 高频 GT：NLL finetune

```bash
python3 BiasDy/mainEurocPLATO.py \
  --plato_supervision gt_se23 \
  --plato_stage nll \
  --outputdir results/Euroc_PLATO_gt_nll_w64 \
  --resume_gyro results/Euroc_PLATO_gt_mse_w64/Gyro_weights/model_epoch_1800.pt \
  --resume_acc results/Euroc_PLATO_gt_mse_w64/Acc_weights/model_epoch_1800.pt \
  --plato_window_steps 64 \
  --batch_size 64 \
  --epoch_nll 300 \
  --lr_NLL 1e-4
```

### 低频长窗口 VO：MSE

```bash
python3 BiasDy/mainEurocPLATO.py \
  --plato_supervision vo_pose \
  --plato_stage mse \
  --outputdir results/Euroc_PLATO_vo_mse_5hz_2s \
  --vo_dir /path/to/vo/results \
  --vo_template "{seq}/00_estimate.txt" \
  --vo_align first_gt \
  --plato_window_steps 400 \
  --plato_sup_stride 40 \
  --batch_size 64 \
  --epoch_mse 1801
```

### 低频长窗口 VO：NLL finetune

```bash
python3 BiasDy/mainEurocPLATO.py \
  --plato_supervision vo_pose \
  --plato_stage nll \
  --outputdir results/Euroc_PLATO_vo_nll_5hz_2s \
  --vo_dir /path/to/vo/results \
  --vo_template "{seq}/00_estimate.txt" \
  --vo_align first_gt \
  --resume_gyro results/Euroc_PLATO_vo_mse_5hz_2s/Gyro_weights/model_epoch_1800.pt \
  --resume_acc results/Euroc_PLATO_vo_mse_5hz_2s/Acc_weights/model_epoch_1800.pt \
  --plato_window_steps 400 \
  --plato_sup_stride 40 \
  --batch_size 16 \
  --epoch_nll 300 \
  --lr_NLL 1e-4
```

---

## 9. 验收标准

实现完成后，仓库应满足：

1. 原始 `mainEuroc.py` 仍可跑 baseline。
2. 新 `mainEurocPLATO.py` 可跑 `gt_se23` 和 `vo_pose` 两种 supervision。
3. MSE 阶段严格保持先 gyro、后 freeze gyro 学 acc。
4. 第一阶段 loss 使用 `log(X_ref^{-1} X_pred)` 的 SE2(3)/pose Lie residual，而不是 raw matrix MSE 或 raw `v,p` MSE。
5. 第二阶段 NLL 使用 block-tridiagonal precision 和离散化 covariance propagation。
6. 不显式形成 dense marginal covariance `P_s`。
7. 不改变任何原始网络结构。
8. 输出 TensorBoard scalar 至少包括：

   * `train/se23_mse_rot`
   * `train/se23_mse_vel`
   * `train/se23_mse_pos`
   * `train/nll`
   * `train/log_sigma_a`
   * `train/log_sigma_g`
   * validation loss
9. 每个 checkpoint 记录：

   * supervision mode
   * window steps
   * supervision stride/rate
   * measurement covariance
   * learned IMU covariance
   * code stage: `mse_gyro`, `mse_acc`, or `nll`

---
