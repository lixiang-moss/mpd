# DMEval（串行版）

DMEval 是一个**与规划器解耦**、可扩展的评估框架，用于运动规划实验的统一运行与结果聚合。

本仓库版本强调：
- **串行调度**（不做复杂并行/异步）
- **指标只做提取，不做计算**：轨迹指标数值由规划器（当前示例是 MPD）内部计算，DMEval 只负责加载/抽取/汇总/导出
- **接口化适配**：通过 Adapter 机制对接不同规划器/推理器

---
## 关键设计点：指标“从哪里来”

以 MPD 为例，指标计算在 MPD 内部完成，然后保存到结果文件里：
- 指标计算：`mpd-splines-public/mpd/metrics/metrics.py`（`PlanningMetricsCalculator.compute_metrics`）
- 推理产物保存：`mpd-splines-public/scripts/inference/inference.py`（保存 `results_single_plan-*.pt`，其中包含 `metrics`）

DMEval 的 MPD 适配器只做**读取 `.pt` + 读取 `args_inference.yaml` + 提取字段**：
- `mpd-splines-public/dmeval/src/dmeval/adapters/mpd.py`

---
## Docker/宿主机环境自查说明（你提到的重点）

我在宿主机环境中做过一些“依赖探测式”的导入检查（例如 `torch/pandas` 等缺失），这些失败是**环境缺包**导致的，而不是把它们当作代码错误去修。
（当前版本 dmeval 已将 `hydra-core` 作为核心依赖。）

为避免“宿主机路径/环境”影响 Docker 使用，我做了两点修正/规避：
- 示例配置不再写死宿主机绝对路径（`workdir: .`），见 `mpd-splines-public/dmeval/configs/example_mpd_jobs.yaml`
- DMEval 对“缺少可选依赖”会给出明确报错信息（例如收集 `.pt` 结果时如果没装 `torch` 会提示安装 `dmeval[mpd]`）

本项目中我没有执行真正的 MPD 推理（GPU/IsaacGym 等），因此不存在“某条命令在宿主机跑不通→误判代码错误”的情况。

---
## 在 Docker 中需要补的依赖（建议）

### 1）只使用 `dmeval run`（仅编排运行，不收集 `.pt`）
- `python>=3.8`
- `pip`
- `PyYAML`
- `hydra-core`（配置管理；安装 dmeval 会一并安装）

安装（在 `mpd-splines-public/` 目录下）：
```bash
python -m pip install -e dmeval
```

### 2）使用 `dmeval collect` 读取 MPD 的 `.pt` 结果（推荐）
- 需要 1）里的全部
- 还需要：`torch`、`dotmap`（用于 `torch.load` 与读取 DotMap 结构）

安装：
```bash
python -m pip install -e "dmeval[mpd]"
```

### 3）使用 `dmeval plot` 画基础对比图（可选）
- 需要：`matplotlib`

安装：
```bash
python -m pip install -e "dmeval[plot]"
```

> 注：MPD 本身运行所需依赖（IsaacGym、scipy、sklearn 等）不在 DMEval 的职责范围内，仍按你现有 MPD Docker 环境来配置。

---
## 快速开始（MPD 示例）

### 1）先运行 MPD 推理（DMEval 只负责编排，不替代 MPD）

例如：
```bash
python3 mpd-splines-public/scripts/inference/inference.py \
  --cfg_inference_path mpd-splines-public/scripts/inference/cfgs/config_EnvSpheres3D-RobotPanda_00.yaml \
  --results_dir mpd-splines-public/scripts/inference/logs/dmeval_demo/unipc_steps20_order2 \
  --seed 2 \
  --save_results_single_plan_low_mem true \
  --device cuda:0
```

MPD（通过 `experiment_launcher`）会写出：
- `<results_dir>/<seed>/results_single_plan-*.pt`
- `<results_dir>/<seed>/args_inference.yaml`

### 2）用 DMEval 聚合成 CSV（并输出排名表）

```bash
dmeval collect \
  --adapter mpd \
  --results_root mpd-splines-public/scripts/inference/logs/dmeval_demo \
  --out_dir mpd-splines-public/scripts/inference/logs/dmeval_demo/_dmeval
```

输出：
- `trial_metrics.csv`：每个 `results_single_plan-*.pt` 一行（trial 级）
- `run_metrics.csv`：按 `(scenario, run_tag, seed)` 聚合后的均值/方差（run 级）
- `rank_*.csv`：基于 run 级均值的排序结果

### 3）画基础图（可选）

```bash
dmeval plot \
  --run_metrics mpd-splines-public/scripts/inference/logs/dmeval_demo/_dmeval/run_metrics.csv \
  --out_dir mpd-splines-public/scripts/inference/logs/dmeval_demo/_dmeval/plots
```

---
## `dmeval run`（串行任务编排）

示例配置见：
- `mpd-splines-public/dmeval/configs/example_mpd_jobs.yaml`

执行：
```bash
dmeval run --config mpd-splines-public/dmeval/configs/example_mpd_jobs.yaml
```

说明：
- DMEval 会按 `jobs` 与 `seeds` **逐个串行执行**命令
- 支持在命令里使用占位符：`{results_root}`、`{run_tag}`、`{seed}`

---
## 架构与接口文档（中文）

请阅读：
- `mpd-splines-public/dmeval/docs/架构与接口.md`

---
## Stage I/II（调参与对比）工作流（符合 proposal01 的两阶段范式）

说明：`example_mpd_tune.yaml` / `example_mpd_compare.yaml` 使用 Hydra `defaults` 组合配置（见 `mpd-splines-public/dmeval/configs/scenario/`、`objective/`、`planner/`），并支持用 `--override key=value` 覆盖。

### Stage I：单场景调参（网格搜索 + 目标函数）

```bash
dmeval tune --config mpd-splines-public/dmeval/configs/example_mpd_tune.yaml
```

示例：覆盖场景与试验次数（仍然串行）：
```bash
dmeval tune --config mpd-splines-public/dmeval/configs/example_mpd_tune.yaml \
  --override scenario=warehouse \
  --override common_inference_args.n_start_goal_states=20
```

产物（示例路径）：
- 调参运行结果：`<pipeline_root>/stage1_tune/<scenario_name>/...`
- 最优配置输出目录：`<pipeline_root>/stage1_tune/<scenario_name>/_dmeval/best_configs/<sampler>/best.yaml`
- 复用用的 sampler patch：`.../best_patch.yaml`（用于 Stage II 跨场景套用）

### Stage II：用最优配置跨场景对比

```bash
dmeval compare --config mpd-splines-public/dmeval/configs/example_mpd_compare.yaml
```

产物（示例路径）：
- 对比运行结果：`<pipeline_root>/stage2_compare/<scenario>/<sampler>/<seed>/...`
- 聚合与排名：`<pipeline_root>/stage2_compare/_dmeval/run_metrics.csv`、`rank_*.csv`
