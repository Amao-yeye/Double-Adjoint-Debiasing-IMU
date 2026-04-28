import torch
import lie_algebra as Lie

def SO3log_loss(R_pred, R_gt, loss_type = torch.nn.MSELoss()):
   residual = Lie.SO3log(R_pred.transpose(-1,-2) @ R_gt)
   return loss_type(residual, torch.zeros_like(residual))

def SE23_loss(X_pred, X_gt, loss_type = torch.nn.MSELoss()):
   """SE_2(3) invariant error loss.
   X_pred, X_gt: (..., 5, 5) SE_2(3) matrices
   Returns MSE of the 9-dim log error xi = [phi, rho_v, rho_p]
   """
   xi = Lie.SEn3log(Lie.SEn3inverse(X_gt) @ X_pred)
   return loss_type(xi, torch.zeros_like(xi))

def SE23_decay_loss(X_pred, X_gt, dt, decay_lambda=1.0):
   """SE_2(3) loss with exponential time decay weights.
   X_pred, X_gt: (batch, window, 5, 5)
   dt: time step size
   decay_lambda: decay rate, w(k) = exp(-decay_lambda * k * dt)
   """
   xi = Lie.SEn3log(Lie.SEn3inverse(X_gt) @ X_pred)
   N = xi.shape[-2]
   k = torch.arange(N, device=xi.device, dtype=xi.dtype)
   weights = torch.exp(-decay_lambda * k * dt)
   weights = weights / weights.sum() * N
   return (weights.unsqueeze(-1) * xi ** 2).mean()

def SE23_nll_loss(
   X_pred,
   X_gt,
   P,
   measurement_cov=None,
   H=None,
   jitter=1e-6,
   skip_initial=0,
   reduction="mean",
   include_logdet=True,
):
   """Gaussian NLL for SE_2(3) residuals in invariant error coordinates.

   Residual order is xi = [theta, v, p], matching the 9x9 covariance propagated
   by the paper right-invariant error dynamics. For full-state supervision H is
   the identity and can be left as None. If H is provided, it maps the full
   invariant state error to the supervisory residual: r ~= H xi.
   """
   xi = Lie.SEn3log(Lie.SEn3inverse(X_gt) @ X_pred)
   P = P.to(device=xi.device, dtype=xi.dtype)
   if P.shape[-2:] != (9, 9):
      raise ValueError(f"P should have shape (..., 9, 9), got {tuple(P.shape)}")
   if skip_initial < 0:
      raise ValueError("skip_initial should be non-negative")
   if skip_initial:
      xi = xi[skip_initial:]
      P = P[skip_initial:]

   if H is None:
      S = P
      r = xi
   else:
      H = H.to(device=xi.device, dtype=xi.dtype)
      if skip_initial and H.ndim >= 3 and H.shape[0] == skip_initial + xi.shape[0]:
         H = H[skip_initial:]
      if H.shape[-1] != 9:
         raise ValueError(f"H should have shape (..., residual_dim, 9), got {tuple(H.shape)}")
      r = (H @ xi.unsqueeze(-1)).squeeze(-1)
      S = H @ P @ H.transpose(-1, -2)

   if measurement_cov is not None:
      measurement_cov = measurement_cov.to(device=xi.device, dtype=xi.dtype)
      if skip_initial and measurement_cov.ndim >= 3 and measurement_cov.shape[0] == skip_initial + xi.shape[0]:
         measurement_cov = measurement_cov[skip_initial:]
      if measurement_cov.shape[-2:] != S.shape[-2:]:
         raise ValueError(
            f"measurement_cov should have trailing shape {tuple(S.shape[-2:])}, "
            f"got {tuple(measurement_cov.shape[-2:])}"
         )
      S = S + measurement_cov

   eye = torch.eye(S.shape[-1], device=xi.device, dtype=xi.dtype)
   S = 0.5 * (S + S.transpose(-1, -2)) + jitter * eye

   chol = torch.linalg.cholesky(S)
   whitened = torch.cholesky_solve(r.unsqueeze(-1), chol).squeeze(-1)
   mahalanobis = (r * whitened).sum(dim=-1)

   if include_logdet:
      logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
      nll = 0.5 * (mahalanobis + logdet)
   else:
      nll = 0.5 * mahalanobis

   if reduction == "mean":
      return nll.mean()
   if reduction == "sum":
      return nll.sum()
   if reduction == "none":
      return nll
   raise ValueError("reduction should be one of: mean, sum, none")
