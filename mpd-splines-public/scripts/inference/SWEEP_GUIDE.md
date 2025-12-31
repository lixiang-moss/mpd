# 采样器扫参使用说明（DPM-Solver / UniPC）

> 目的：说明“DPM‑Solver/++ 扫参脚本”与“UniPC 扫参脚本”的使用方式、实验矩阵思路与输出位置。

## 1) DPM‑Solver/++ 更全面扫参（推荐）

**脚本：** `mpd-splines-public/scripts/inference/run_dpm_solver_best_sweep.py`  
**用途：** 用“两阶段”顺序扫参，尽量少跑冤枉路地找出 DPM‑Solver/++ 的较优配置。  

### 两阶段思路（为什么这么设计）
你会发现很多配置都 `success=1`（只要 100 条里有 1 条有效就算成功），这时用 0/1 很难区分优劣。  
所以这个脚本的策略是：

1) **Coarse（粗扫）**：先只扫最关键、影响最大的维度（步数/阶数/连续或离散 + solver vs solver++），快速找到“稳定且质量不错”的区域。  
2) **Refine（细扫）**：只在 coarse 的 Top‑K 赢家上，再扫少量“内部开关”（method/skip/denoise_to_zero/lower_order_final…），把算力集中在更可能提升的地方。  

### 评价指标（脚本用哪些来选 Top‑K）
对每个 `run_tag`（可包含多个 seed）会做聚合（平均，忽略 NaN），并按以下优先级排序：
1. `success_mean`（越大越好）
2. `fraction_valid_mean`（越大越好：有效轨迹比例）
3. `path_length_best_mean`（越小越好）
4. `smoothness_best_mean`（越小越好）
5. `t_inference_total_mean`（越小越好）

### 使用方式（推荐直接跑 all）
```bash
python3 mpd-splines-public/scripts/inference/run_dpm_solver_best_sweep.py \
  --base_cfg mpd-splines-public/scripts/inference/cfgs/config_EnvSpheres3D-RobotPanda_00.yaml \
  --planner_alg mpd \
  --phase all \
  --collect_metrics \
  --device cuda:0
```

### 默认实验矩阵（你不改参数时会跑什么）
**Coarse：**（默认比较 solver vs solver++）
- steps：`15,20,26,32,40`
- order：`2,3`
- 时间评估：离散 + 连续（`use_continuous_time: false/true`）
- solver vs solver++：通过 `dpm_solver`（dpmsolver） vs `dpm_solver_pp`（dpmsolver++）对比

**Refine：**（默认对 coarse Top‑K=3 做细扫）
- `dpm_solver_method`：`multistep,singlestep`
- `dpm_solver_skip_type`：`logSNR`
- `dpm_solver_solver_type`：`dpmsolver`
- `lower_order_final`：`true,false`
- `dpm_solver_denoise_to_zero`：`false,true`
  
**可选（如果你想把 guide 相关参数也纳入最优搜索）：**
- `--refine_prior_weights`（对应 `prior_weight_with_guide`）
- `--refine_n_guide_steps`（对应 `n_guide_steps`）
- `--refine_guide_lr`（对应 `guide_lr`）
- `--refine_max_perturb_x`（对应 `max_perturb_x`）
- `--refine_t_start_guide_steps_fraction`（对应 `t_start_guide_steps_fraction`）
  
> 以上可选项默认为“空”，表示使用 base_cfg 里的固定值，不参与 sweep。

> 设备一次只能跑一个实验：脚本是顺序执行（不并行），中断后可用 `--skip_existing` / `--skip-existing`（默认开启）断点续跑；用 `--no_skip_existing` / `--no-skip-existing` 关闭跳过。

---

## 2) DPM‑Solver 快速 sweep（顺序跑，旧脚本）

**脚本：** `mpd-splines-public/scripts/inference/run_mpd_sweep.sh`  
**用途：** 只扫 steps/continuous（非常快），适合做基线或 sanity check。  

### 使用方式
```bash
bash mpd-splines-public/scripts/inference/run_mpd_sweep.sh
```

### 默认行为
- sampler：`dpm_solver`
- planner：`mpd`
- 步数：`15,20,26,32,40`
- 阶数：固定 `order=2`
- 时间模式：离散 + 连续（`use_continuous_time: false/true`）
- 顺序执行（不并行）

### 可调整参数
编辑 `mpd-splines-public/scripts/inference/run_mpd_sweep.sh`：
- `BASE_CFG`：基础配置文件
- `DEVICE`：GPU/CPU 设备
- steps/order/time_modes 等会传给 `run_sampler_sweep.py`

---

## 3) UniPC 扫参（顺序跑）

**脚本：** `mpd-splines-public/scripts/inference/run_sampler_sweep.py`  
**用途：** 生成配置并顺序执行；支持 UniPC 的 steps/order/continuous/variant 扫参。  

### 使用方式（示例）
```bash
python3 mpd-splines-public/scripts/inference/run_sampler_sweep.py \
  --base_cfg mpd-splines-public/scripts/inference/cfgs/config_EnvSpheres3D-RobotPanda_00.yaml \
  --method unipc \
  --planner_alg mpd \
  --steps 20,40 \
  --orders 1,2,3 \
  --time_modes both \
  --variants bh1 \
  --collect_metrics \
  --device cuda:0
```

**只生成配置不运行：**
```bash
python3 mpd-splines-public/scripts/inference/run_sampler_sweep.py \
  --base_cfg mpd-splines-public/scripts/inference/cfgs/config_EnvSpheres3D-RobotPanda_00.yaml \
  --method unipc \
  --generate_only
```

---

## 4) 输出文件与记录位置

### 4.1 生成的配置文件
- 位置：`mpd-splines-public/scripts/inference/cfgs/generated/`
- 命名：`<base_cfg_name>__<run_tag>.yaml`

### 4.2 运行结果目录

- 默认根目录（可通过 `--results_root` 覆盖）：
  - `run_sampler_sweep.py`：`mpd-splines-public/scripts/inference/logs/sweep/`
  - `run_dpm_solver_best_sweep.py`：`mpd-splines-public/scripts/inference/logs/sweep_dpm_solver/`
- 每次运行会生成：
  - `<run_tag>/<seed>/args_inference.yaml`（实际使用配置）
  - `<run_tag>/<seed>/results_single_plan-000.pt`（核心结果）
  - `<run_tag>/<seed>/logfile`（日志）
  - 可选视频：如果开启渲染相关选项，会在同目录生成 mp4

> 注意：experiment_launcher 会把 `results_dir` 自动加上 `seed` 子目录。
>
> **省磁盘提示（重要）：** 这两个 sweep 脚本会在调用 `inference.py` 时默认启用
> `--save_results_single_plan_low_mem true`，也就是只保存“计时 + metrics + best trajectory”等必要信息，
> 避免把完整的扩散迭代链（iters）写入 `results_single_plan-000.pt` 导致磁盘爆炸。

### 4.3 指标汇总 CSV

汇总脚本：`mpd-splines-public/scripts/inference/collect_sweep_metrics.py`  
当你使用 `--collect_metrics` 参数时，会自动生成：

- `sweep_metrics.csv`（完整指标表）
- `sweep_rank_success.csv`（按成功率降序）
- `sweep_rank_fraction_valid.csv`（按有效轨迹比例降序）
- `sweep_rank_path_length.csv`（按路径长度升序）
- `sweep_rank_speed.csv`（按推理总时间升序）

`sweep_metrics.csv` 还会包含（用于更细的质量/安全对比）：
- `collision_intensity`
- `ee_pose_goal_error_*`（best / all-mean-std / valid-mean-std）

输出位置：你传入的 `--results_root/`（例如默认的 `mpd-splines-public/scripts/inference/logs/sweep/`）。

---

## 5) 结果字段：哪些最值得保留？

扫参/对比采样器时，**最有意义（建议必保留）**：
- 速度：`t_inference_total`（以及 `t_generator` / `t_guide`，方便判断瓶颈在采样还是在 guide）
- 稳定性：`metrics.trajs_all.fraction_valid`（比 0/1 的 `success` 更有区分度）
- 质量（最终会执行的那条）：`metrics.trajs_best.path_length`、`metrics.trajs_best.smoothness`、`metrics.trajs_best.ee_pose_goal_error_*`
- 安全/碰撞：`metrics.trajs_all.collision_intensity`（如果你关心碰撞，这是关键）

通常可不保留（或保留也无所谓）：
- `isaacgym_statistics`（未开启 IsaacGym 评估时经常是空的 `DotMap()`）
- `*_no_joint_limits_vel_acc` 这组指标（排查失败原因时有用，但不是主排序指标）

## 5) 关键指标说明（轨迹质量）

这些指标来自 `results_single_plan-000.pt` 的 `metrics` 字段：

- `trajs_all.success`：成功（0/1：只要至少 1 条有效轨迹就算成功）  
- `trajs_all.fraction_valid`：有效轨迹占比（0~1：例如 0.68 表示 100 条里约 68 条有效）  
- `trajs_valid.path_length_mean/std`：有效轨迹路径长度  
- `trajs_best.path_length`：最佳轨迹路径长度  
- `trajs_valid.smoothness_mean/std`：平滑度  
- `t_inference_total / t_generator / t_guide`：耗时

---

## 6) 常见问题

- **跑一次只跑一个实验吗？**  
  是的，这些脚本默认顺序执行（不并行），适合单机。

- **如何查看某次运行的指标？**  
  使用 `collect_sweep_metrics.py` 汇总，或直接读取 `results_single_plan-000.pt`。
