Current baseline definition:
The existing baseline is the RSS Debiasing baseline with window size w = 64 and the bg / ba checkpoint. 
In all following experiments, "baseline" refers to this method unless otherwise specified.

Overall goal:
Complete and compare two types of supervision for inertial-only raw preintegration and OpenVINS evaluation:
1. Dense GT supervision
2. Sparse VO supervision

All reported improvements should be computed as relative error reduction:
improvement (%) = (baseline_error - method_error) / baseline_error * 100.
Lower rotation and position errors are better.

============================================================
Part I: Raw Preintegration Experiments
============================================================

Goal:
Evaluate the raw inertial preintegration performance using learned bias dynamics under GT-supervised and VO-supervised training.

网络结构也需要修改：
1. 你可以采用Euroc数据集中我们结果最优的结构
2. 也可以采用如下建议：dot b = -lambda * b + s * tanh(MLP(b, t, stats(u_{t-0.02:t})))

  具体配置：

  phi(u) = 0.02s causal IMU window statistics
  net = MLP-64
  lambda = 0.1 1/s
  s_g = 0.02 rad/s^2
  s_a = 0.10 m/s^3
  lr = 0.01

  也就是：

  dot b = -0.1 * b + s * tanh(f_theta(b, t, phi(u_{t-20:t})))


下述任务可以通过修改网络input, 网络结构等来实现。


----------------------------------------------------------------
Experiments to complete:

1. GT-MSE
   - Method name: ours_gt_w64_se23_mse
   - Use dense ground-truth trajectory supervision.
   - Use the strict SE_2(3) MSE training version.
   - Target: outperform the baseline by at least 8% in position and 5% in rotation.

2. GT-NLL
   - Method name: ours_gt_w64_se23_nll
   - Initialize from the GT-MSE model.
   - Fine-tune using the NLL objective.
   - Target: outperform GT-MSE by at least 3% in both rotation error and position error.

3. VO-MSE
   - Method name: ours_vo_w64_se23_mse
   - Use sparse VO supervision.
   - The VO supervision should cover a longer time span than dense GT supervision since it is more sparse.
   - Use the strict SE_2(3) MSE training version.
   - Target: outperform the baseline by at least 8% in position and 5% in rotation.

4. VO-NLL
   - Method name: ours_vo_w64_se23_nll
   - Initialize from the VO-MSE model.
   - Fine-tune using the NLL objective.
   - Target: outperform VO-MSE by at least 3% in both rotation error and position error.

Required output for raw preintegration:
- A table comparing baseline, GT-MSE, GT-NLL, VO-MSE, and VO-NLL.
- Report rotation error and position error for each sequence and the mean.
- Highlight whether each method satisfies the target improvement.
- Save trained checkpoints and logs with clear names matching the method names above.

============================================================
Part II: OpenVINS Experiments
============================================================




Goal:
Evaluate whether the learned bias compensation and learned covariance improve OpenVINS performance.

Step 1: Ours OpenVINS
Run the following four variants:

1. GT-MSE + OpenVINS
   - Use the bias model trained by GT-MSE.
   - Use default covariance settings in the OpenVINS config:
     n_a = 1e-6, n_g = 1e-6.
   - Target: outperform the OpenVINS baseline by at least 5% in both rotation error and position error.

2. GT-NLL + OpenVINS
   - Use the bias model fine-tuned by GT-NLL.
   - Use the covariance learned by the NLL objective in the OpenVINS config.
   - Target: outperform GT-MSE + OpenVINS by at least 5% in both rotation error and position error.
   - This experiment should demonstrate that the learned covariance is beneficial.

3. VO-MSE + OpenVINS
   - Use the bias model trained by VO-MSE.
   - Use default covariance settings in the OpenVINS config:
     n_a = 1e-6, n_g = 1e-6.
   - Target: outperform the OpenVINS baseline by at least 5% in both rotation error and position error.

4. VO-NLL + OpenVINS
   - Use the bias model fine-tuned by VO-NLL.
   - Use the covariance learned from VO-NLL supervision in the OpenVINS config.
   - Target: outperform the OpenVINS baseline by at least 7% in both rotation error and position error.

Required output for OpenVINS:
- A table comparing:
  baseline OpenVINS,
  GT-MSE OpenVINS,
  GT-NLL OpenVINS,
  VO-MSE OpenVINS,
  VO-NLL OpenVINS.
-  openvins报告中，需要专门报告drop out段，raw, baseline，和ours。
- Report rotation error and position error for each sequence and the mean.
- Clearly indicate which covariance setting is used for each method:
  default covariance or NLL-learned covariance.
- Highlight whether each method satisfies the target improvement.

============================================================
Implementation Notes
============================================================

1. Keep the window size fixed at w = 64 for all methods.
2. Use the same train / test split across all comparable experiments.
3. Use the same evaluation metrics and scripts across baseline and our methods.
4. For MSE variants, use the learned bias model only and keep the OpenVINS covariance fixed at the default values.
5. For NLL variants, use both the NLL-fine-tuned bias model and the learned covariance parameters.
6. Save all checkpoints, logs, evaluation results, and OpenVINS config files with unambiguous names.
7. Ensure that every result table includes both per-sequence results and the mean result.
8. After running each experiment, compute the relative improvement automatically and print a pass / fail flag for the target requirement.

Final deliverables:
1. Raw preintegration result table.
2. OpenVINS result table.
3. Checkpoints for all trained models.
4. Logs for all training and evaluation runs.
5. OpenVINS config files used for each method.
6. A short summary stating which targets are satisfied and which targets still need improvement.

