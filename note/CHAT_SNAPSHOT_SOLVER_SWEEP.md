# 对话快照（后半段）— Solver/UniPC 扫参、pw 含义、低内存日志与图表化

> 适用仓库：`/home/woss/MPDLX-B-new`  
> 只关注推理（inference），可忽略训练相关内容。  
> 目的：给另一个对话窗口/另一个 ChatGPT 快速“接上上下文”。
>
> 场景标注约定（避免误用）：`logs/` 下**路径名包含 `replace`** 的 sweep/结果属于**场景02**；不包含 `replace` 的属于**场景01**。两种场景的数据只做“各自内部”对比，别混到同一张表/同一张图里直接排序。

---

## 1) 我们在做什么（阶段目标）

1. **对比/扫参采样器**：主要关注 `DPM-Solver++`（`dpm_solver_pp`）与 `UniPC` 的推理表现。  
2. **解决“连续时间 vs 离散时间”严格性问题**：高阶采样器（DPM-Solver/UniPC）理论上是连续时间（或 logSNR 对齐）的多步法；如果把连续 `t` 强行 `round()/int()`，会退化成分段常数近似。  
3. **磁盘爆炸问题**：原本每次推理保存大量中间链（iters / 全量轨迹集合），扫参会爆盘；改成 “low-mem 只保留必要信息”。  
4. **输出可读性**：把关键指标集中写入 `sweep_metrics.csv`，并提供“自动图表化/聚合排序”的脚本。

---

## 2) 关键脚本/入口（你需要知道的文件）

### 2.1 扫参脚本
- 通用扫参（含 UniPC）：`mpd-splines-public/scripts/inference/run_sampler_sweep.py`
  - 用途：生成一批 `cfgs/generated/*.yaml` 并逐个调用 `scripts/inference/inference.py` 跑推理。
- DPM-Solver(++）专用扫参：`mpd-splines-public/scripts/inference/run_dpm_solver_best_sweep.py`
  - 用途：按“coarse → refine”思路扫 steps/order/time_mode 以及（可选）guide 侧参数。

### 2.2 汇总 CSV（扫完后把结果变成表格）
- `mpd-splines-public/scripts/inference/collect_sweep_metrics.py`
  - 读取每个 run 的结果目录，汇总为 `sweep_metrics.csv` + 一些 `sweep_rank_*.csv`。

### 2.3 图表化（把 CSV 变成图 + 聚合表）
- `mpd-splines-public/scripts/inference/plot_sweep_metrics.py`
  - 输入：`sweep_metrics.csv`
  - 输出：`plots/aggregated_by_run_tag.csv`、`plots/top_configs.txt`、`plots/*.png`

---

## 3) 扫参输出都在哪里看？

### 3.1 你最近的 solver sweep（低内存版）
- 结果根目录：`mpd-splines-public/scripts/inference/logs/sweep_dpm_solver_stage1_lowmem_v1/`
- 原始逐条结果表：`mpd-splines-public/scripts/inference/logs/sweep_dpm_solver_stage1_lowmem_v1/sweep_metrics.csv`
- 图表与聚合：`mpd-splines-public/scripts/inference/logs/sweep_dpm_solver_stage1_lowmem_v1/plots/`
  - `aggregated_by_run_tag.csv`：跨 seeds 按 run_tag 聚合后的表（便于筛选/排序）
  - `top_configs.txt`：按 `(fraction_valid_min, fraction_valid_mean, time)` 排序的 Top 列表
  - `scatter_fraction_min_vs_time.png` 等：用来直观挑配置

### 3.2 UniPC sweep（之前的矩阵实验）
- 结果根目录：`mpd-splines-public/scripts/inference/logs/sweep_unipc_big/`

---

## 4) “pw / prior_weight_with_guide” 到底是什么？（为什么 DDIM 里是 0.25）

### 4.1 在这个项目里，pw 的数学意义
当启用 guide（代价梯度引导）时，把扩散模型输出整体乘一个系数：
- DDPM：`noise_pred = prior_weight_with_guide * noise_pred`
- DPM-Solver：`eps_pred = prior_weight_with_guide * eps_pred`
- UniPC：`model_out = prior_weight_with_guide * model_out`

直觉：**pw 越大 = 相对更信“模型先验”**；pw 越小 = 相对更信“guide 梯度”。

### 4.2 为什么 DDIM 的 0.25 不能和 pw=1.05 直接比？
DDIM 里叫 `ddim_scale_grad_prior`，它缩放的是 “prior 梯度项”，在组合公式中的位置不同（不是同一个参数语义）。

---

## 5) 你这次 solver sweep（stage1_lowmem_v1）跑了什么？

核心：固定一套 solver++ 形态，只扫 steps + guide 侧组合。

### 5.1 固定项（本次 sweep 没有在这些维度上比较）
- sampler：`dpm_solver_pp`（solver++）
- `use_continuous_time = true`（ct=1）
- `dpm_solver_order = 2`
- `dpm_solver_method = multistep`
- `dpm_solver_skip_type = logSNR`
- `dpm_solver_solver_type = dpmsolver`
- `lower_order_final = false`
- `dpm_solver_denoise_to_zero = true`

### 5.2 扫的变量（全排列）
- `dpm_solver_steps ∈ {16, 26, 40, 60, 80}`
- `prior_weight_with_guide (pw) ∈ {0.85, 0.95, 1.05}`
- `guide_lr ∈ {0.01, 0.03, 0.06, 0.10, 0.30, 1.00}`
- `t_start_guide_steps_fraction ∈ {0.05, 0.10, 0.20, 0.30}`
- `n_guide_steps ∈ {1, 2, 3}`
- `max_perturb_x ∈ {0.05, 0.10, 0.15, 0.20}`
- `seeds ∈ {0,1,2}`

---

## 6) 基于 stage1 的结论（在 A 场景内的“经验最优区间”）

从 `plots/top_configs.txt` 看到的共性（面向“成功率/可行性”为主）：
- `pw=1.05` 显著优于 `0.95/0.85`
- `guide_lr` 极敏感：`0.01` 最好，`0.03` 次之；`>=0.06` 明显变差
- `n_guide_steps=1` 基本够用（2/3 主要是变慢）
- `t_start_guide_steps_fraction` 与 steps 有耦合：高 steps（60/80）时 `0.2` 更优；低 steps（26/40）时 `0.3` 往往更好
- `max_perturb_x` 在最优区域不太“卡脖子”（不如 pw/guide_lr 敏感）
- steps：`80` 最强，`60` 更快且几乎不掉（可做默认）

---

## 7) 连续时间泛化担忧（“A 训练 + A 测试”是否会误导 pw 结论？）

你的担心是合理的：**只在训练分布（同一环境/障碍分布）上选超参，得到的是“in-distribution 最优”，不保证 out-of-distribution 仍最优**。

论文 `Motion Planning Diffusion (arXiv:2308.01557)` 明确做了“Extra Obstacles（训练没见过的新障碍）”测试，并指出：
- 只采 prior（不够强的引导）在未见障碍上成功率会下降（属于预期的分布外退化）
- 用 cost guidance 可以把性能拉回来

**对应到本项目：pw（先验强弱）与 guide 强度确实可能在分布外需要重新平衡。**

### 7.1 不会“造新场景”怎么办？
在本项目里仍有可行的“验证集/分布外近似”：
1) **用 validation start/goal**：`scripts/inference/inference.py` 支持 `selection_start_goal: validation`（默认就是 validation）  
2) **改障碍位置（你说的“只改位置”）**：`EnvSpheres3D` 支持 `spheres_center_noise`，给球心加随机扰动（属于“新场景但同类型”）  
3) **加额外障碍**：有 `EnvSpheres3DExtraObjectsV00`（属于更强的分布外）

思路：拿少量候选配置（比如 Top-5 + pw=0.95 的对照）在 (validation contexts) × (多个障碍扰动强度/extra obstacles) × (多 seeds) 上复测，选“最稳健”的，而不是只选 A 内最优。

---

## 8) 常用命令（最短路径）

### 8.1 重新生成图表与聚合 CSV
```bash
python3 mpd-splines-public/scripts/inference/plot_sweep_metrics.py \
  --results_root mpd-splines-public/scripts/inference/logs/sweep_dpm_solver_stage1_lowmem_v1
```

### 8.2 直接看“最优配置列表”
- `mpd-splines-public/scripts/inference/logs/sweep_dpm_solver_stage1_lowmem_v1/plots/top_configs.txt`

---

## 9) 下一步建议（最省实验量的路线）

1) 先把 stage1 的最优区域固定（比如 steps=60/80、glr=0.01、ng=1、tsg=0.2、pw=1.05）  
2) 再做“小规模分布外验证”：`spheres_center_noise` 从 0 → 0.05 → 0.10（每档少量 contexts + 多 seeds）  
3) 如果分布外掉得厉害，再考虑把 pw/guide_lr/tsg 作为“鲁棒性旋钮”重新扫一个更小的网格
