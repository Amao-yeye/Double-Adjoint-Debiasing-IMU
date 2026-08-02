import torch
import torch.nn as nn


def compute_acc_history_rate_features(times, acc, history):
    """Return causal [acc, rate_K] features with baseline-like 9D total input."""
    if history <= 0:
        raise ValueError("history must be positive")
    if times.dim() != 1:
        raise ValueError("times must have shape [T]")
    if acc.dim() != 2 or acc.shape[-1] != 3:
        raise ValueError("acc must have shape [T, 3]")
    if acc.shape[0] != times.shape[0]:
        raise ValueError("times and acc must have the same length")

    n = acc.shape[0]
    idx = torch.arange(n, device=acc.device)
    start_idx = (idx - history + 1).clamp_min(0)
    elapsed = (times - times[start_idx]).unsqueeze(-1)
    median_dt = torch.median(times[1:] - times[:-1]).clamp_min(1e-6)
    elapsed = torch.where(elapsed > 0, elapsed, median_dt.expand_as(elapsed))
    rate_k = (acc - acc[start_idx]) / elapsed
    return torch.cat([acc, rate_k], dim=-1)


class AccHistoryFeatures:
    def __init__(self, times, acc, history, device="cpu", eps=1e-9):
        self._times = times.to(device)
        self._features = compute_acc_history_rate_features(
            self._times,
            acc.to(device),
            history,
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


class ba_func_net_history_rate(nn.Module):
    """Accel-bias dynamics using [b_a, acc, rate_K], total input dim 9."""

    uses_acc_history = True
    history_feature_kind = "rate"

    def __init__(self, feature_dim=6):
        super().__init__()
        self.feature_dim = feature_dim
        input_dim = 3 + feature_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, input_dim),
        )
        self.net2 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, input_dim),
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

    def forward(self, y, imu_meas, imu_meas_dot, R0, acc_history=None):
        if acc_history is None:
            raise ValueError("ba_func_net_history_rate requires acc_history features")
        mean = self.feature_mean.to(acc_history.device, acc_history.dtype)
        std = self.feature_std.to(acc_history.device, acc_history.dtype)
        features = (acc_history - mean) / std
        x = torch.cat([y[..., 13:16], features], dim=-1)
        x = self.net(x) + x
        x = self.net2(x) + x
        return self.linear2(x)


def compute_acc_history_normalizer(dataset, history, device):
    features = []
    for index in range(dataset.length()):
        imudict = dataset.load_imu(index)
        times = imudict["t_imu"].to(device)
        acc = imudict["u_imu"][:, 3:].to(device)
        feat = compute_acc_history_rate_features(times, acc, history)
        train_end = round(feat.shape[0] * (1 - dataset._percent_for_val))
        features.append(feat[:train_end])
    all_features = torch.cat(features, dim=0)
    return all_features.mean(dim=0), all_features.std(dim=0, unbiased=False).clamp_min(1e-6)


def set_acc_history_func_for_sequence(model, dataset, index, history, device):
    if history <= 0:
        model.set_acc_history_func(None)
        return None
    imudict = dataset.load_imu(index)
    control = AccHistoryFeatures(
        imudict["t_imu"].to(device),
        imudict["u_imu"][:, 3:].to(device),
        history,
        device=device,
    )
    model.set_acc_history_func(control.evaluate)
    return control
