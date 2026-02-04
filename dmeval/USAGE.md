# 使用说明（DMEval vNext / L1）

> 目标：用同一份配置跑通闭环：**Stage I tune → best_configs → Stage II compare → CSV/plots**。  
> 说明：DMEval 自身只负责“编排运行 + 抽取/聚合/比较”，指标数值由被测系统（例如 MPD）产出。

## 0. 目录与入口

- 代码：`dmeval/src/dmeval/`
- 默认配置：`dmeval/conf/config.yaml`
- CLI（推荐）：`python -m dmeval.cli ...`（需要把 `dmeval/src` 加到 `PYTHONPATH`）

## 1. 依赖（容器里需要安装/具备）

DMEval 侧（最小）：
- `python>=3.9`
- `hydra-core>=1.3`
- `omegaconf`
- `PyYAML`
- `pandas`
- `matplotlib`

如果你只解析 MPD 的 `.pt` 结果（没有 `trial_metrics.jsonl`），还需要：
- `torch`（用于 `torch.load`）

> 你当前 MPD 的 conda 环境文件 `mpd-splines-public/environment.yml` 已经包含 `pandas / omegaconf / PyYAML` 等依赖；若缺 `hydra-core`/`matplotlib`，在对应环境里补装即可。

## 2. 快速自检（不跑 MPD，10 秒内验证闭环）

仓库自带一个 dummy planner：`dmeval/scripts/dummy_planner.py`，会输出 `trial_metrics.jsonl` 来模拟 MPD 的产物结构。

在仓库根目录执行（示例）：

```bash
python3 -m venv .tmp/dmeval-venv
. .tmp/dmeval-venv/bin/activate
python -m pip install -U pip
python -m pip install hydra-core omegaconf PyYAML pandas matplotlib

PYTHONPATH=dmeval/src \
python -m dmeval.cli --config dmeval/conf/config.yaml \
  -o planner=dummy \
  -o pipeline.root=.tmp/dmeval_demo \
  -o pipeline.allow_overwrite=true \
  -o common_inference_args.n_start_goal_states=3 \
  run
```

检查输出：
- `.tmp/dmeval_demo/tune/*/trial_metrics.csv`
- `.tmp/dmeval_demo/best_configs/*/best_patch.yaml`
- `.tmp/dmeval_demo/compare/run_metrics.csv`
- `.tmp/dmeval_demo/compare/plots/*.png`

## 3. 用 MPD 跑最小端到端样例（论文闭环）

### 3.1 前置：确认 MPD 可单独跑通一次 inference

以 MPD 自带场景为例（你也可以换成自己的 base cfg）：
- `mpd-splines-public/scripts/inference/cfgs/config_EnvSpheres3D-RobotPanda_00.yaml`
- `mpd-splines-public/scripts/inference/cfgs/config_EnvPlanar2Link-RobotPlanar2Link_00.yaml`

先单跑一条（确认环境/模型/数据路径没问题）：

```bash
cd mpd-splines-public
python3 scripts/inference/inference.py \
  --cfg_inference_path scripts/inference/cfgs/config_EnvPlanar2Link-RobotPlanar2Link_00.yaml \
  --results_dir /tmp/mpd_smoke \
  --seed 2 \
  --selection_start_goal validation \
  --n_start_goal_states 1 \
  --save_results_single_plan_low_mem true \
  --device cpu
```

### 3.2 运行 DMEval：Stage I + Stage II

1) 复制并修改配置（推荐新建自己的 config，避免覆盖）：

```bash
cp dmeval/conf/config.yaml dmeval/conf/config_mpd.yaml
```

2) 在 `dmeval/conf/config_mpd.yaml` 里根据你的需求修改：
- `pipeline.root`：输出目录（建议每次用新目录）
- `common_inference_args.device`：`cuda:0` 或 `cpu`
- `tune.scenario.base_cfg`：Stage I 用的单场景 base cfg
- `tune.samplers`：要调参的 sampler 列表与 search space
- `compare.scenarios`：Stage II 对比的多个场景 base cfg
- `tune.seeds / compare.seeds` 与 `n_start_goal_states`

3) 执行闭环（同一进程，保证 `pipeline.root` 一致）：

```bash
PYTHONPATH=dmeval/src \
python -m dmeval.cli --config dmeval/conf/config_mpd.yaml \
  -o planner=mpd \
  run
```

### 3.3 输出解读（你写论文时会用到）

Stage I（每个 sampler 一套）：
- `<pipeline.root>/tune/<sampler>/trial_metrics.csv`：trial 级别（每行一个 start-goal）
- `<pipeline.root>/tune/<sampler>/run_metrics.csv`：run 级别（按 seed 聚合）
- `<pipeline.root>/tune/<sampler>/candidate_metrics.csv`：candidate 级别（跨 seed/trials 聚合）
- `<pipeline.root>/best_configs/<sampler>/best_patch.yaml`：Stage II 复用的最优 patch
- `<pipeline.root>/best_configs/<sampler>/topk.yaml`：top-k 备查
- `<pipeline.root>/tune_manifest.yaml`：完整记录（含 resolved config）

Stage II：
- `<pipeline.root>/compare/trial_metrics.csv`
- `<pipeline.root>/compare/run_metrics.csv`
- `<pipeline.root>/compare/run_metrics_agg.csv`：按 (scenario,sampler) 聚合（用于公平比较）
- `<pipeline.root>/compare/rank_*.csv`：成功率/有效率/路径长度/速度排名
- `<pipeline.root>/compare/plots/*.png`：基础图（success/fraction_valid/time + time vs success）

## 4. 常见坑（基于你的项目环境）

- **pipeline.root 每次要固定**：如果你把 tune 和 compare 分开两条命令跑，需要显式 `-o pipeline.root=...` 保持一致；否则默认 `${now:...}` 会变成两个目录。
- **MPD 没有 trial_metrics.jsonl 时**：DMEval 会回退解析 `.pt`，这要求环境里能 import `torch` 并能 `torch.load`。
- **路径不要写死**：MPD 入口脚本/工作目录都在 `dmeval/conf/planner/mpd.yaml` 里配置，不要在代码里改。

