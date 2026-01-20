
# dpm-solver options
dpm_solver:
  t_start_guide_steps_fraction: 0.3
  dpm_solver_steps: 10
  dpm_solver_order: 2
  dpm_solver_method: 'multistep'  # 'multistep', 'singlestep', 'singlestep_fixed'
  dpm_solver_algorithm_type: 'dpmsolver++'
  dpm_solver_skip_type: 'time_uniform'  # 'time_uniform', 'logSNR', 'time_quadratic'
  dpm_solver_solver_type: 'dpmsolver'  # 'dpmsolver', 'taylor'
  dpm_solver_denoise_to_zero: false
  use_continuous_time: false


DPM-Solver++ (Lu 等人，2022c) 指出了先验高阶求解器的一个关键局
限性：它们存在稳定性问题，并且在大引导尺度（更强的条件）下可能比
DDIM更慢。作者将这种不稳定性归因于大引导尺度对输出及其导数的放大
作用。由于高阶求解器依赖于高阶导数，因此它们对这种效应特别敏感，导
致效率和稳定性下降。