# 项目快照（推理阶段）

> 目的：在开启新对话时，快速恢复上下文与当前状态。  
> 范围：**仅推理 (inference)**；训练/数据生成流程不在本快照讨论范围内。

## 0. 项目概览

- 工作区根目录：`/home/woss/MPDLX-B-new`
- 主要子工程：
  - `mpd-splines-public/`：MPD（Motion Planning Diffusion）主框架与推理脚本
  - `dpm_solver/`：DPM-Solver 参考实现（推理时按需导入）
  - `unipc/`：UniPC 参考实现（推理时按需导入）
  - `data_public/`：示例/公开数据（推理用数据与模型通常通过软链接或环境变量引用）

---

## 1. 核心功能进度

### 1.1 已实现（推理相关）

- **MPD 推理链路（已有）**
  - 通过 `mpd-splines-public/scripts/inference/cfgs/*.yaml` 配置推理参数（环境、代价函数、采样器、引导参数等）。
  - `mpd-splines-public/scripts/inference/inference.py` 作为推理入口（基于 `experiment_launcher` 解析 CLI 参数）。
  - 推理时加载预训练模型：`args_inference.model_dir/checkpoints/model_current.pth`（由配置中的 `model_dir_*` 指定）。

- **采样器（已有）**
  - `ddpm`：DDPM 采样循环
  - `ddim`：DDIM 采样循环（包含可选的引导梯度更新）
  - `dpm_solver` / `dpm_solver_pp`：DPM-Solver / DPM-Solver++（以 wrapper 方式接入 MPD，并复用 MPD 的 hard conditioning + guidance 钩子）

- **UniPC 采样器接入（本阶段新增）**
  - 新增 `unipc` 作为可选采样方法，可在 YAML 中通过 `diffusion_sampling_method: 'unipc'` 切换。
  - 与现有采样器 **解耦**：独立文件实现，不改动 DDPM/DDIM/DPM-Solver 内部逻辑。
  - 支持 MPD 的：
    - hard conditioning（每步投影固定条件点）
    - cost guide（梯度引导，复用 `guide_gradient_steps`）
    - 计时统计（`results_ns.t_generator / t_guide`）

- **采样器 sweep（扫参）基础设施（本阶段新增）**
  - DPM‑Solver/++：`mpd-splines-public/scripts/inference/run_dpm_solver_best_sweep.py`（两阶段 coarse→refine）
  - UniPC（以及旧的 DPM-Solver 快速扫）：`mpd-splines-public/scripts/inference/run_sampler_sweep.py`
  - sweep 运行后自动汇总指标到 CSV：`mpd-splines-public/scripts/inference/collect_sweep_metrics.py`
  - sweep 使用说明：`mpd-splines-public/scripts/inference/SWEEP_GUIDE.md`

- **省磁盘保存（low-mem）用于大规模 sweep（本阶段新增）**
  - 推理入口 `mpd-splines-public/scripts/inference/inference.py` 新增/支持 `save_results_single_plan_low_mem`：
    - **开启时**：`results_single_plan-000.pt` 只保存“计时 + metrics + 少量复现信息 + best trajectory”，避免保存完整扩散迭代链导致磁盘爆炸。
  - sweep 脚本默认开启 low-mem（不需要你手动加参数）：
    - `run_dpm_solver_best_sweep.py`、`run_sampler_sweep.py` 都会调用 `inference.py --save_results_single_plan_low_mem true`

- **sweep 指标更细化（本阶段新增）**
  - `sweep_metrics.csv` 除了 `success/fraction_valid/path/smooth/time` 外，还会额外写入：
    - `collision_intensity`
    - `ee_pose_goal_error_*`（best / all-mean-std / valid-mean-std）
  - 用于直接排序/筛选（不用再手动去 `.pt -> metrics` 里翻字段）。

### 1.2 待办（推理相关）

- **数值/理论一致性验证**
  - 对 UniPC 的输出质量、收敛稳定性、与 DDIM/DPM-Solver 的对比评估（同一环境/同一随机种子/同一 guide 设置）。
- **关键原则偏差的修复评估**
  - 当前 UniPC wrapper 在“连续时间 t → 离散索引”的映射上使用了取整（详见第 5 节），严格性不足；需要决定是否做最小改动修复。
- **推理可复现性**
  - 增加一份“最小可运行命令 + 环境变量检查”说明（便于新机器/新容器复现）。

---

## 2. 技术栈与架构

### 2.1 技术栈（推理侧）

- Python（项目期望 Python3；注意系统默认 `python` 可能指向 Python2，需要显式使用 `python3` 或 conda 环境）
- PyTorch（推理/采样/梯度引导）
- IsaacGym（仿真/环境，推理脚本中直接 `import isaacgym`）
- `torch_robotics/`（项目内置子模块/依赖，提供机器人模型、环境、计时器等）
- 关键第三方库：`numpy`, `einops`, `dotmap`

### 2.2 推理架构（关键组件）

- 配置层：
  - `mpd-splines-public/scripts/inference/cfgs/*.yaml`
  - 通过 `DotMap(load_params_from_yaml(...))` 载入，形成 `args_inference`
- 推理入口：
  - `mpd-splines-public/scripts/inference/inference.py`
  - 调用 `mpd.inference.inference.GenerativeOptimizationPlanner` 执行轨迹生成与（可选）代价引导
- 生成模型：
  - `mpd-splines-public/mpd/models/diffusion_models/diffusion_model_base.py`（`GaussianDiffusionModel`）
  - `conditional_sample(..., method=...)` 根据 `diffusion_sampling_method` 选择采样器实现
- 采样器模块化：
  - DDPM/DDIM：在 `diffusion_model_base.py`/`sample_functions.py` 内
  - DPM-Solver：`mpd-splines-public/mpd/models/diffusion_models/dpm_solver_sampler.py`
  - UniPC：`mpd-splines-public/mpd/models/diffusion_models/unipc_sampler.py`（本阶段新增）

---

## 3. 核心数据结构（推理侧关键变量/契约）

### 3.1 配置与运行时对象

- `args_inference`：从 YAML 读取的 `DotMap`，包含：
  - `diffusion_sampling_method`：`'ddpm' | 'ddim' | 'dpm_solver' | 'dpm_solver_pp' | 'unipc'`
  - `planner_alg`：`'diffusion_prior' | 'diffusion_prior_then_guide' | 'mpd' | ...`
  - `n_trajectory_samples`：采样 batch size
  - `costs`：代价项与权重（用于引导）
- `results_ns`：计时/结果记录对象（`DotMap`），用于累计 `t_generator`、`t_guide` 等。

### 3.1.1 结果保存文件（sweep 关心的契约）

- 每次推理输出目录：`<results_root>/<run_tag>/<seed>/`
  - `args_inference.yaml`：实际运行时的配置快照（用于复现）
  - `results_single_plan-000.pt`：核心结果（由 `save_results_single_plan_low_mem` 控制体积）
  - `logfile`：推理日志

- `save_results_single_plan_low_mem=true` 时，`results_single_plan-000.pt` **顶层**只保证包含：
  - 计时：`t_inference_total / t_generator / t_guide`
  - `metrics`（包含 `success/fraction_valid/path/smoothness/collision/ee_error/...` 等汇总指标）
  - 少量复现上下文：`q_pos_start / q_pos_goal / ee_pose_goal`
  - 便于 debug 的最优轨迹：`control_points_best / q_trajs_*_best`
  - 不保存：完整扩散链的 `*_iters` 与全量 `*_valid` 轨迹集合（这些是 sweep 磁盘爆炸的根源）

### 3.2 采样与条件输入

- `shape_x`：采样张量形状，典型为 `(B, horizon, state_dim)`。
- `hard_conds`：字典 `{time_index: state_tensor}`，由 `apply_hard_conditioning(x, hard_conds)` 每步强制写入。
- `context_d`：上下文输入字典（由 dataset 构建），用于 `context_model` 编码成 `context_emb`。

### 3.3 扩散模型关键缓存

- `GaussianDiffusionModel` 中注册的 buffers（推理用）：
  - `betas`, `alphas_cumprod`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`, `posterior_variance`, ...
  - `predict_epsilon`：决定模型输出语义（预测噪声 ε 还是预测 x0）

### 3.4 采样器配置键（推理用）

> 以下键来自 `scripts/inference/cfgs/*.yaml` 的各采样器小节，并通过 `mpd/inference/inference.py` 转为 `sample_fn_kwargs`。

- `ddpm:` 常用键：`t_start_guide_steps_fraction`, `prior_weight_with_guide`, `noise_std`, `n_guide_steps`, `guide_lr`, ...
- `ddim:` 常用键：`ddim_sampling_timesteps`, `ddim_eta`, `ddim_skip_type`, `ddim_scale_grad_prior`, `n_guide_steps`, `guide_lr`, ...
- `dpm_solver:` 常用键：`dpm_solver_steps`, `dpm_solver_order`, `dpm_solver_skip_type`, `dpm_solver_algorithm_type`, `dpm_solver_method`, ...
- `unipc:`（新增）常用键：
  - `unipc_steps`：步数
  - `unipc_order`：阶数上限（含 warm-up 的阶数提升逻辑）
  - `unipc_skip_type`：**强制 `logSNR`**（log-SNR 对齐）
  - `unipc_variant`：`'bh1' | 'bh2' | 'vary_coeff'`
    - `bh1`：UniPC‑BH 变体，系数归一化采用 `B(h)=h`（见 `mpd-splines-public/mpd/models/diffusion_models/unipc_sampler.py` 中 `B_h = hh`）
    - `bh2`：UniPC‑BH 变体，系数归一化采用 `B(h)=exp(h)-1`（见 `mpd-splines-public/mpd/models/diffusion_models/unipc_sampler.py` 中 `B_h = expm1(hh)`）
    - `vary_coeff`：UniPC “varying coefficients” 变体，每步用 `r_k` 构造矩阵并求逆得到系数（更通用，但每步多一次小矩阵求逆/解线性方程）
  - `unipc_lower_order_final`：末端降阶
  - `unipc_denoise_to_zero`：末端额外去噪到 0（可选）
  - 以及通用引导参数：`t_start_guide_steps_fraction`, `prior_weight_with_guide`, `n_guide_steps`, `guide_lr`, `max_perturb_x`, `clip_grad*`, ...

---

## 4. 重要决策记录（ADR）

### ADR-001：UniPC 以“独立 sampler 模块”方式接入

- 决策：新增 `mpd/models/diffusion_models/unipc_sampler.py`，并在 `GaussianDiffusionModel.conditional_sample` 里用 method 路由到 UniPC。
- 原因：
  - 保证与 DDPM/DDIM/DPM-Solver **解耦**，降低互相干扰风险。
  - 复用 MPD 已有的 hard conditioning 与 cost guidance 逻辑，避免重复实现。

### ADR-002：UniPC 步长强制使用 log-SNR（λ）对齐

- 决策：UniPC 在推理侧强制 `unipc_skip_type == 'logSNR'` 并做断言。
- 原因：
  - 避免误把离散 t 当作与 log-SNR 线性对应，从而破坏 UniPC 的关键假设。

### ADR-003：解析线性项 + 显式广播作为“数学护栏”

- 决策：UniPC 更新里显式使用解析形式的线性项（涉及 `exp`, `expm1`, `h_phi`），并对时间系数做显式 reshape 广播。
- 原因：
  - 避免用通用 Euler 近似直接处理线性项。
  - 避免隐式广播导致的 silent bug（尤其是 `(B,)` vs `(B,...)`）。

### ADR-004：sweep 默认启用 low-mem 保存，避免磁盘爆炸

- 决策：在 `inference.py` 中增加 `save_results_single_plan_low_mem`，并在 sweep 脚本中默认开启。
- 原因：
  - sweep 场景只需要“计时 + metrics（+ 可选最优轨迹）”，不需要保存完整扩散链与全量样本集合。
  - 否则 `.pt` 很容易累计到数百 GB。

---

## 5. 待解决的 Bug / 坑点 / 风险

### 5.1 UniPC 严格性风险：连续时间模型评估被“取整索引化”

- 现状：UniPC wrapper 中，连续时间 `t_continuous` 会被映射为离散 index（round），再喂给模型。
- 风险：
  - 高阶连续时间多步法依赖“在不同时间点评估模型输出的变化趋势”；取整后模型输出在时间上变成分段常数（台阶），会削弱甚至部分抵消高阶优势。
- 规避方案（不改代码）：
  - 若追求严格一致性：推理优先使用 `ddim` 或 `dpm_solver`（它们与离散训练时间更一致）。
  - 若坚持用 `unipc`：建议 `unipc_order` 取 1~2（降低高阶依赖），并避免把 `unipc_steps` 拉得过大导致大量步落在同一离散 index。
- 后续若允许改动（最小修复方向）：让模型输入时间保持连续（不取整），例如用连续 `t_input`（与模型时间嵌入兼容）。

### 5.2 环境坑：系统默认 `python` 可能是 Python2

- 现状：系统 `python --version` 可能返回 2.7；项目推理需要 Python3 + PyTorch。
- 建议：统一用 `python3 ...` 或进入 conda 环境后运行。

### 5.3 依赖/版本风险

- UniPC 实现使用 `torch.linalg.inv/solve`，需要较新 PyTorch 版本。
- IsaacGym/驱动与 GPU 环境不匹配会导致推理脚本导入失败。
- `collect_sweep_metrics.py` 依赖 `torch` 去读 `.pt`；必须在装了 PyTorch 的推理环境中运行（通常与运行 `inference.py` 同一个 conda env）。

---

## 6. 推理使用方式（Quickstart）

### 6.1 运行推理

```bash
cd mpd-splines-public/scripts/inference
python3 inference.py --cfg_inference_path ./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml
```

### 6.2 启用 UniPC

在对应 YAML 中设置：

```yaml
diffusion_sampling_method: 'unipc'
```

并在 `unipc:` 小节里调整 `unipc_steps / unipc_order / unipc_variant` 等参数。

### 6.3 DPM‑Solver/++ 扫参（推荐）

脚本：`mpd-splines-public/scripts/inference/run_dpm_solver_best_sweep.py`  
说明文档：`mpd-splines-public/scripts/inference/SWEEP_GUIDE.md`

```bash
python3 mpd-splines-public/scripts/inference/run_dpm_solver_best_sweep.py \
  --base_cfg mpd-splines-public/scripts/inference/cfgs/config_EnvSpheres3D-RobotPanda_00.yaml \
  --planner_alg mpd \
  --phase all \
  --collect_metrics \
  --device cuda:0
```

输出位置（默认）：`mpd-splines-public/scripts/inference/logs/sweep_dpm_solver/`  
关键汇总文件：`sweep_metrics.csv` 与 `sweep_rank_*.csv`

### 6.4 UniPC 扫参（对比离散/连续 + order/steps）

脚本：`mpd-splines-public/scripts/inference/run_sampler_sweep.py`  

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

### 6.5 重新汇总 CSV（当你更新了汇总逻辑或新增列）

```bash
python3 mpd-splines-public/scripts/inference/collect_sweep_metrics.py \
  --results_root <你的 sweep 结果根目录>
```

---

## 7. 已跑 sweep 的结论（推理侧，基于现有日志）

> 说明：以下结论来自你项目内已存在的 sweep 结果目录；如果你新增/继续跑了更多组合，需要重新跑一次 `collect_sweep_metrics.py` 才会反映到 CSV。

### 7.1 数据来源（你要复查时看这里）

- DPM‑Solver/++（完整两阶段 sweep 的历史结果）：
  - 目录：`mpd-splines-public/scripts/inference/logs/sweep_dpm_solver_full/`
  - 关键文件：`sweep_metrics.csv`、`sweep_rank_fraction_valid.csv` 等
- UniPC（对比离散/连续 + order/steps + variant；当前只跑了 seed=2）：
  - 目录：`mpd-splines-public/scripts/inference/logs/sweep_unipc_big/`
  - 关键文件：`sweep_metrics.csv`、`sweep_rank_fraction_valid.csv` 等

### 7.2 DPM‑Solver/++ 的结论（以 `fraction_valid` + 质量为主）

- **更推荐用 `dpm_solver_pp`（DPM‑Solver++）**：同等/更少步数下整体更稳、质量更好；且 sweep 的优胜配置几乎都来自 `dpm_solver_pp`。
- **`denoise_to_zero=true` 是“质变级”开关**：在最优区间（例如 steps≈40, order≈2）里，开启后 `fraction_valid` 和 `smoothness` 都显著改善。
- **`use_continuous_time`（ct1）总体是“轻微正收益，但不稳定”**：
  - 在 coarse 网格里，ct1 相对 ct0 的 `fraction_valid` 提升幅度非常小（约 0.1~0.4 个百分点量级），不同 seed 的波动更大。
  - 在历史最优区域里，ct1 出现频率更高（在 `sweep_dpm_solver_full` 的综合排序 Top‑15 里 ct1 多于 ct0）。
  - 由于模型时间嵌入是正弦（见 `mpd-splines-public/mpd/models/layers/layers.py` 的 `SinusoidalPosEmb`），**传入浮点时间是可行的**；是否收益取决于训练分布/guide 强度等。

**一个可作为默认起点的 solver++ 配置（来自 `sweep_dpm_solver_full` 的多 seed 聚合赢家之一）：**
- `diffusion_sampling_method: dpm_solver_pp`
- `dpm_solver_steps: 40`
- `dpm_solver_order: 2`
- `use_continuous_time: true`
- `dpm_solver_method: multistep`
- `dpm_solver_skip_type: logSNR`
- `dpm_solver_solver_type: dpmsolver`
- `dpm_solver_denoise_to_zero: true`
- `lower_order_final: false`（在已跑结果中通常影响很小）

### 7.3 UniPC 的结论（当前为“初步结论”，因只跑 1 个 seed）

- **步数是主导因素**：在 `sweep_unipc_big` 里，`unipc_steps` 从 20/26/32 提升到 80/100 后，`fraction_valid` 才能接近 0.44~0.45。
- **`use_continuous_time`（ct1）在当前 sweep 中没有体现出稳定优势**：
  - ct1 与 ct0 的平均 `fraction_valid` 几乎一样（且在不同 steps/order/variant 上有正有负）。
  - 这更像是“需要多 seed 才能下结论”的信号，而不是证明 ct 无效。
- **variant 的差异较小**：在最高 `fraction_valid` 的区域（80~100 steps）里，`bh1/bh2/vary_coeff` 都能达到相近水平；`vary_coeff` 在少数配置上有略好的 `path/smoothness`，但差异不如 “steps” 明显。
- **与 solver++ 的对比**：在当前日志下（尤其是 `fraction_valid` 指标），UniPC 整体仍明显落后于 `dpm_solver_pp` 的最佳区域；如果后续要认真比较，建议把 UniPC 也做多 seed + 聚焦高 steps 的二阶段 sweep。

---

## 8. 本阶段改动文件清单（推理侧）

- 新增：`mpd-splines-public/mpd/models/diffusion_models/unipc_sampler.py`
- 修改：
  - `mpd-splines-public/mpd/models/diffusion_models/diffusion_model_base.py`（增加 `method == 'unipc'` 分支）
  - `mpd-splines-public/mpd/inference/inference.py`（增加 `diffusion_sampling_method == 'unipc'` 的配置处理）
  - `mpd-splines-public/scripts/inference/launch_inference-experiments.py`（sweep 增加 `unipc`）
  - `mpd-splines-public/scripts/inference/cfgs/*.yaml`（增加 `unipc:` 配置段与注释）
  - `mpd-splines-public/scripts/inference/inference.py`（增加/使用 `save_results_single_plan_low_mem`，用于 sweep 省磁盘）
  - `mpd-splines-public/scripts/inference/run_sampler_sweep.py`（默认启用 low-mem 保存；用于 UniPC/旧 sampler sweep）
  - `mpd-splines-public/scripts/inference/run_dpm_solver_best_sweep.py`（DPM‑Solver/++ 两阶段扫参脚本；默认启用 low-mem 保存）
  - `mpd-splines-public/scripts/inference/collect_sweep_metrics.py`（汇总 sweep 指标并写 CSV；增加 collision/ee_error 列）
  - `mpd-splines-public/scripts/inference/SWEEP_GUIDE.md`（扫参使用说明与指标解释）
