


1. Set up a new devcontainer, using the original MPD project's Conda environment configuration as the base.


2. Integrated the solver-series samplers (DPM‑Solver / DPM‑Solver++) into MPD, enabling switching via configuration.   (e.x. `/home/woss/MPDLX-B-new/mpd-splines-public/scripts/inference/cfgs/config_EnvSpheres3D-RobotPanda_00.yaml`).

3. Developed and utilized scripts to batch-generate configurations and execute the experiment matrix, with inference results automatically exported to CSV files for comparison. Currently using this to screen for the optimal solver configuration (in progress).

4. Attempted to integrate UniPC; while it is currently usable for inference, only the initial interface binding has been completed.