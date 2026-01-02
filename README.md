


1. Set up a new devcontainer, using the original MPD project's Conda environment configuration as the base.


2. Integrated the solver-series samplers (DPM‑Solver / DPM‑Solver++) into MPD, enabling switching via configuration.   (e.x. `/home/woss/MPDLX-B-new/mpd-splines-public/scripts/inference/cfgs/config_EnvSpheres3D-RobotPanda_00.yaml`).

3. Developed and utilized scripts to batch-generate configurations and execute the experiment matrix, with inference results automatically exported to CSV files for comparison. Currently using this to screen for the optimal solver configuration (in progress).

4. Attempted to integrate UniPC; while it is currently usable for inference, only the initial interface binding has been completed.

## Git LFS assets (missing objects)

Some files are Git LFS pointers but the actual LFS objects are not present(these LFS only in github).

Affected path (example):
- mpd-splines-public/mpd/torch_robotics/torch_robotics/data/urdf/robots/habitat_stretch/meshes/

Note:
- Current inference configs in this project (e.g. RobotPanda / RobotPlanar2Link / RobotPointMass2D) do NOT use these `habitat_stretch` mesh assets.
- These files are only needed if you load the Stretch robot model (e.g. `DifferentiableHabitatStretch` / `hab_stretch.urdf`).



