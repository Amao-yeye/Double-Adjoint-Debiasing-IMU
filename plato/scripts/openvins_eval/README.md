# OpenVINS Evaluation Scripts

PLATO-owned scripts for evaluating learned or corrected IMU bias dynamics in
OpenVINS.

## LIASuw Clean-IMU Route

Export raw, constant-bias-corrected, and network-debiased IMU CSVs:

```bash
/home/amao/anaconda3/envs/debiasimu/bin/python \
  plato/scripts/openvins_eval/export_liasuw_openvins_inputs.py
```

Build OpenVINS-readable ROS bags from those CSVs and the existing LIASuw stereo
image export:

```bash
python3 plato/scripts/openvins_eval/make_liasuw_openvins_bags.py --compression bz2
```

Outputs are under:

```text
results/openvins_eval/LIASuw/phys_delta_vo_framefixed_clean_imu/0205s14_80/
```

See `plato/docs/liasuw_openvins_clean_imu_route.md` for validation details and
OpenVINS runtime notes.

The current LIASuw config keeps OpenVINS' optional learned-bias propagation hook
disabled with `use_learned_bias_dynamics: false`, so these clean-IMU comparisons
continue to use the original OpenVINS nominal bias behavior unless that flag is
explicitly enabled. The config points to the exported PLATO `phys_delta` weights
at `results/debias_imu/LIASuw_phys_delta_e1800_w16_b64/openvins/phys_delta_openvins.yaml`;
regenerate that file with:

```bash
/home/amao/anaconda3/envs/debiasimu/bin/python \
  plato/scripts/openvins_eval/export_phys_delta_to_openvins_yaml.py
```

After OpenVINS has written `runs/gtinit_full_*/estimate.txt`, summarize the
direct GT-init comparison with:

```bash
/home/amao/anaconda3/envs/debiasimu/bin/python \
  plato/scripts/openvins_eval/summarize_liasuw_openvins_runs.py \
  --methods raw raw_constbias constant network
```

To expose propagation behavior under visual failure, build bags that remove both
stereo image streams over a selected interval:

```bash
python3 plato/scripts/openvins_eval/make_visual_dropout_bags.py \
  --methods constant network --dropout-start-s 5 --dropout-end-s 35 \
  --compression bz2
```

Run OpenVINS while recording `/ros1_serial_msckf/odomimu`:

```bash
plato/scripts/openvins_eval/run_openvins_dropout_trial.sh \
  --method constant --dropout-start 5 --dropout-end 35
plato/scripts/openvins_eval/run_openvins_dropout_trial.sh \
  --method network --dropout-start 5 --dropout-end 35
```

Then summarize the dropout comparison with:

```bash
/home/amao/anaconda3/envs/debiasimu/bin/python \
  plato/scripts/openvins_eval/summarize_liasuw_openvins_dropout.py \
  --dropout-start-s 5 --dropout-end-s 35
```

The export script also writes GT helper files under `groundtruth/`. Use
`body_gt_openvins_asl.csv` for the 200 Hz reference, and only use
`body_gt_openvins_asl_gtinit.csv` for controlled OpenVINS GT-initialization
diagnostics.
