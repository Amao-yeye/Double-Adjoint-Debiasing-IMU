import torch
import torch.nn as nn


def compute_gyro_history_features(times, gyro, history):
    """Return causal [gyro, mean_K, std_K, short_rate, long_rate] features."""
    if history <= 0:
        raise ValueError("history must be positive")
    if times.dim() != 1:
        raise ValueError("times must have shape [T]")
    if gyro.dim() != 2 or gyro.shape[-1] != 3:
        raise ValueError("gyro must have shape [T, 3]")
    if gyro.shape[0] != times.shape[0]:
        raise ValueError("times and gyro must have the same length")

    n = gyro.shape[0]
    padded = torch.cat([gyro[:1].expand(history - 1, -1), gyro], dim=0)
    windows = torch.stack([padded[start : start + n] for start in range(history)], dim=1)
    gyro_mean = windows.mean(dim=1)
    gyro_std = windows.std(dim=1, unbiased=False)

    delta_short = torch.zeros_like(gyro)
    dt_prev = (times[1:] - times[:-1]).unsqueeze(-1).clamp_min(1e-6)
    delta_short[1:] = (gyro[1:] - gyro[:-1]) / dt_prev

    idx = torch.arange(n, device=gyro.device)
    start_idx = (idx - history + 1).clamp_min(0)
    elapsed = (times - times[start_idx]).unsqueeze(-1)
    median_dt = torch.median(times[1:] - times[:-1]).clamp_min(1e-6)
    elapsed = torch.where(elapsed > 0, elapsed, median_dt.expand_as(elapsed))
    delta_long = (gyro - gyro[start_idx]) / elapsed

    return torch.cat([gyro, gyro_mean, gyro_std, delta_short, delta_long], dim=-1)


def compute_gyro_history_rate_features(times, gyro, history):
    """Return causal [gyro, rate_K] features with baseline-like 9D total input."""
    if history <= 0:
        raise ValueError("history must be positive")
    if times.dim() != 1:
        raise ValueError("times must have shape [T]")
    if gyro.dim() != 2 or gyro.shape[-1] != 3:
        raise ValueError("gyro must have shape [T, 3]")
    if gyro.shape[0] != times.shape[0]:
        raise ValueError("times and gyro must have the same length")

    n = gyro.shape[0]
    idx = torch.arange(n, device=gyro.device)
    start_idx = (idx - history + 1).clamp_min(0)
    elapsed = (times - times[start_idx]).unsqueeze(-1)
    median_dt = torch.median(times[1:] - times[:-1]).clamp_min(1e-6)
    elapsed = torch.where(elapsed > 0, elapsed, median_dt.expand_as(elapsed))
    rate_k = (gyro - gyro[start_idx]) / elapsed
    return torch.cat([gyro, rate_k], dim=-1)


def _compute_features_by_kind(times, gyro, history, feature_kind):
    if feature_kind == "full":
        return compute_gyro_history_features(times, gyro, history)
    if feature_kind == "rate":
        return compute_gyro_history_rate_features(times, gyro, history)
    raise ValueError(f"unknown gyro history feature kind: {feature_kind}")


class GyroHistoryFeatures:
    def __init__(self, times, gyro, history, device="cpu", eps=1e-9, feature_kind="full"):
        self._times = times.to(device)
        self._features = _compute_features_by_kind(
            self._times,
            gyro.to(device),
            history,
            feature_kind,
        )
        self._eps = eps

    def evaluate(self, t):
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=self._times.device, dtype=self._times.dtype)
        else:
            t = t.to(device=self._times.device, dtype=self._times.dtype)

        if t.dim() == 0:
            index = torch.searchsorted(self._times, t + self._eps, right=True) - 1
            index = index.clamp(0, self._times.shape[0] - 1)
            return self._features[index]
        if t.dim() == 1:
            index = torch.searchsorted(self._times, t + self._eps, right=True) - 1
            index = index.clamp(0, self._times.shape[0] - 1)
            return self._features[index]
        raise ValueError("t must be a scalar tensor or a 1D tensor")


class bw_func_net_history(nn.Module):
    """Gyro-bias dynamics using [b_g, causal gyro history features]."""

    uses_gyro_history = True
    history_feature_kind = "full"

    def __init__(self, feature_dim=15):
        super().__init__()
        self.feature_dim = feature_dim
        input_dim = 3 + feature_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, input_dim),
        )
        self.net2 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, input_dim),
        )
        self.linear2 = nn.Linear(input_dim, 3)
        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_std", torch.ones(feature_dim))

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.001)
                nn.init.constant_(module.bias, 0)

    def set_feature_normalizer(self, mean, std):
        if mean.shape[-1] != self.feature_dim or std.shape[-1] != self.feature_dim:
            raise ValueError(f"history feature normalizer should have dim {self.feature_dim}")
        self.feature_mean.copy_(mean.detach().to(self.feature_mean.device, self.feature_mean.dtype))
        self.feature_std.copy_(
            std.detach().to(self.feature_std.device, self.feature_std.dtype).clamp_min(1e-6)
        )

    def forward(self, y, imu_meas, imu_meas_dot, R0, gyro_history=None):
        if gyro_history is None:
            raise ValueError("bw_func_net_history requires gyro_history features")
        mean = self.feature_mean.to(gyro_history.device, gyro_history.dtype)
        std = self.feature_std.to(gyro_history.device, gyro_history.dtype)
        features = (gyro_history - mean) / std
        x = torch.cat([y[..., 4:7], features], dim=-1)
        x = self.net(x) + x
        x = self.net2(x) + x
        return self.linear2(x)


class bw_func_net_history_rate(nn.Module):
    """Gyro-bias dynamics using [b_g, omega, rate_K], total input dim 9."""

    uses_gyro_history = True
    history_feature_kind = "rate"

    def __init__(self, feature_dim=6):
        super().__init__()
        self.feature_dim = feature_dim
        input_dim = 3 + feature_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, input_dim),
        )
        self.net2 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, input_dim),
        )
        self.linear2 = nn.Linear(input_dim, 3)
        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_std", torch.ones(feature_dim))

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.001)
                nn.init.constant_(module.bias, 0)

    def set_feature_normalizer(self, mean, std):
        if mean.shape[-1] != self.feature_dim or std.shape[-1] != self.feature_dim:
            raise ValueError(f"history feature normalizer should have dim {self.feature_dim}")
        self.feature_mean.copy_(mean.detach().to(self.feature_mean.device, self.feature_mean.dtype))
        self.feature_std.copy_(
            std.detach().to(self.feature_std.device, self.feature_std.dtype).clamp_min(1e-6)
        )

    def forward(self, y, imu_meas, imu_meas_dot, R0, gyro_history=None):
        if gyro_history is None:
            raise ValueError("bw_func_net_history_rate requires gyro_history features")
        mean = self.feature_mean.to(gyro_history.device, gyro_history.dtype)
        std = self.feature_std.to(gyro_history.device, gyro_history.dtype)
        features = (gyro_history - mean) / std
        x = torch.cat([y[..., 4:7], features], dim=-1)
        x = self.net(x) + x
        x = self.net2(x) + x
        return self.linear2(x)


def compute_gyro_history_normalizer(dataset, history, device, feature_kind="full"):
    features = []
    for index in range(dataset.length()):
        imudict = dataset.load_imu(index)
        times = imudict["t_imu"].to(device)
        gyro = imudict["u_imu"][:, :3].to(device)
        feat = _compute_features_by_kind(times, gyro, history, feature_kind)
        train_end = round(feat.shape[0] * (1 - dataset._percent_for_val))
        features.append(feat[:train_end])
    all_features = torch.cat(features, dim=0)
    return all_features.mean(dim=0), all_features.std(dim=0, unbiased=False).clamp_min(1e-6)


def set_gyro_history_func_for_sequence(model, dataset, index, history, device):
    if history <= 0:
        model.set_gyro_history_func(None)
        return None
    imudict = dataset.load_imu(index)
    feature_kind = getattr(model.biasfunc_w, "history_feature_kind", "full")
    control = GyroHistoryFeatures(
        imudict["t_imu"].to(device),
        imudict["u_imu"][:, :3].to(device),
        history,
        device=device,
        feature_kind=feature_kind,
    )
    model.set_gyro_history_func(control.evaluate)
    return control
