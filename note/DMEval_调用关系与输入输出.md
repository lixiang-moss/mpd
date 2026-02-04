# DMEval 模块调用关系与输入输出（以 MPD 为例）

本文用纯文本（ASCII）把 `abc/dmeval/src/dmeval` 的：
- 文件所属“模块”（按职责分组）
- 文件/函数之间的调用关系（从 CLI 到落盘产物）
- 运行时需要输入什么、会输出什么（以 MPD 适配器为例）

画出来，方便你快速定位“从哪进、到哪出、数据长什么样”。

---
## 1) 文件归属（按模块分组）

```
dmeval/（包：abc/dmeval/src/dmeval）
  [入口层]
  - cli.py                 CLI：解析参数，分发子命令

  [通用基础层]
  - config.py              解析 DMEvalConfig/JobConfig（run 子命令用的 YAML）
  - runner.py              串行执行器：subprocess.run + stdout/stderr 落盘 + run_records
  - utils.py               通用工具：NaN/flatten/token/template 等

  [结果收集层]
  - collect.py             扫描结果树 -> trial_metrics/run_metrics/rank_*.csv
  - plot.py                从 run_metrics.csv 画基础图（matplotlib 可选）

  [编排层：两阶段评估]
  - tune.py                Stage I：生成候选 cfg -> 跑 -> collect -> objective 选 best_configs
  - compare.py              Stage II：读 best_patch -> 多场景 merge -> 跑 -> collect (+plot)

  [契约层：适配不同系统的结果格式]
  - adapters/base.py        ResultsAdapter 协议 + TrialArtifact
  - adapters/mpd.py         MPD 适配器：发现 .pt、读 args_inference、抽取 metrics

  [目标函数层：Stage I 选优策略]
  - objectives/simple.py    可行性 + 打分的示例实现（tune 动态加载）

  [包信息]
  - __init__.py             版本号等
```

---
## 2) 文件级依赖/调用关系总览（静态 import + 主要函数调用）

> 读图方式：`A ──▶ B` 表示 A 调用/依赖 B（import 或运行时调用）。

```
                           ┌──────────────────────┐
                           │      cli.py          │
                           │  main()/argparse     │
                           └───┬───────┬───────┬──┘
                               │       │       │
                               │       │       │
            ┌──────────────────┘       │       └──────────────────┐
            │                          │                          │
            v                          v                          v
      ┌─────────────┐           ┌─────────────┐            ┌─────────────┐
      │  runner.py  │           │ collect.py  │            │   plot.py   │
      │ run_command │           │ collect_*   │            │ plot_*      │
      └──┬───────┬──┘           └──┬───────┬──┘            └─────┬───────┘
         │       │                 │       │                       │
         │       │                 │       │                       │
         │       v                 │       v                       v
         │  ┌──────────┐           │  ┌──────────────┐        ┌──────────┐
         │  │ config.py │           │  │ adapters/mpd │        │ utils.py │
         │  │ load_cfg  │           │  │ + torch/yaml │        └──────────┘
         │  └──────────┘           │  └──────┬───────┘
         │       │                 │         │
         v       v                 v         v
   ┌──────────┐  ┌──────────┐  ┌──────────┐ ┌─────────────┐
   │ utils.py │  │ subprocess│  │ utils.py │ │ adapters/base│
   └──────────┘  │  (外部进程)│  └──────────┘ └─────────────┘
                 └─────┬────┘
                       │
                       v
                ┌──────────────┐
                │ MPD inference │  (外部脚本，产生 .pt + args_inference.yaml)
                └──────────────┘

另外：
- tune.py   ──▶ (runner.py, collect.py, objectives/*, utils.py)
- compare.py ─▶ (runner.py, collect.py, plot.py, utils.py)
```

---
## 3) CLI 子命令的“端到端调用图”（含输入/输出）

### 3.1 `dmeval run`（通用：按 DMEval YAML 跑一组作业）

输入（你需要提供什么）
- CLI：
  - `--config <path>`：DMEval YAML
  - `--dry_run`：只打印命令不执行（可选）
- YAML（由 `config.py:load_config()` 读取）：
  - `results_root`：结果根目录
  - `jobs[]`：每个 job 的 `run_tag/cmd/seeds/...`
  - `cmd` 支持模板变量：`{results_root} {run_tag} {seed}`

输出（会产生什么）
- 结果目录（由你的外部命令决定写什么；dmeval 至少保证每次运行有 stdout/stderr 文件）：
  - `<results_root>/<run_tag>/<seed>/dmeval_stdout.txt`
  - `<results_root>/<run_tag>/<seed>/dmeval_stderr.txt`
- 运行记录：
  - `<results_root>/_dmeval/run_records.yaml`

调用图
```
用户
  │  dmeval run --config dmeval.yaml [--dry_run]
  v
cli.py:main()
  v
runner.py:run_command(config_path, dry_run)
  v
config.py:load_config(dmeval.yaml)  ──▶ (YAML -> DMEvalConfig/JobConfig)
  v
runner.py:run_jobs_serial(DMEvalConfig)
  v
runner.py:_run_one(job, seed)
  v
subprocess.run(cmd, cwd, env, timeout)   (执行外部脚本/二进制)
  v
<results_root>/<run_tag>/<seed>/...      (外部脚本的结果 + dmeval_stdout/stderr)
```

MPD 举例（一个 job 的 cmd 通常长这样）
```yaml
jobs:
  - run_tag: "scene01/ddim"
    seeds: [2, 3]
    workdir: "."
    cmd:
      - "python3"
      - "mpd-splines-public/scripts/inference/inference.py"
      - "--cfg_inference_path"
      - "abc/dmeval/configs/mpd_scene01.yaml"
      - "--results_dir"
      - "{results_root}/{run_tag}"
      - "--seed"
      - "{seed}"
```

---
### 3.2 `dmeval collect`（以 MPD adapter 为例：从结果树提取指标并聚合）

输入
- CLI：
  - `--adapter mpd`（默认 mpd）
  - `--results_root <dir>`：结果根目录（要能递归找到 `results_single_plan-*.pt`）
  - `--out_dir <dir>`：CSV 输出目录
  - `--include_all_config`：把 `args_inference.yaml` 的所有键展平写进 CSV（可选，会很宽）

输出
- `<out_dir>/trial_metrics.csv`：每个 `.pt` 一行（细粒度）
- `<out_dir>/run_metrics.csv`：按 `(scenario, run_tag, seed)` 聚合（mean/std）
- `<out_dir>/rank_*.csv`：基于 run 级别均值做的排名表

MPD 侧结果“契约”（adapter 假设的落盘结构）
```
<results_root>/
  <run_tag>/<seed>/
    args_inference.yaml
    results_single_plan-000.pt
    results_single_plan-001.pt
    ...
```

调用图
```
用户
  │ dmeval collect --adapter mpd --results_root R --out_dir O
  v
cli.py:main()
  v
collect.py:collect_command(...)
  v
collect.py:collect_results(adapter="mpd", results_root=R, out_dir=O)
  v
collect.py:_get_adapter("mpd") ──▶ adapters/mpd.py:MpdResultsAdapter
  v
MpdResultsAdapter.discover_trials(R)  ──▶ yield TrialArtifact（每个 .pt）
  v
MpdResultsAdapter.load_trial_object(artifact)  ──▶ torch.load(.pt)
  v
MpdResultsAdapter.extract_row(artifact, trial_object)
  v
collect.py:_write_csv(trial_metrics.csv)
  v
collect.py:_group_trials(...) ──▶ nanmean/nanstd 聚合
  v
collect.py:_write_csv(run_metrics.csv + rank_*.csv)
```

---
### 3.3 `dmeval plot`（从 run_metrics.csv 画图）

输入
- CLI：
  - `--run_metrics <path/to/run_metrics.csv>`
  - `--out_dir <dir>`
  - `--scenario <name>`：可选，只画某个场景
- 依赖：`matplotlib`（未安装会提示安装 `dmeval[plot]`）

输出
- `<out_dir>/*.png`（柱状图/散点图）

调用图
```
用户
  │ dmeval plot --run_metrics run_metrics.csv --out_dir plots/
  v
cli.py:main()
  v
plot.py:plot_command(...)
  v
plot.py:plot_from_run_metrics(...)
  v
matplotlib 生成图片（落盘到 out_dir）
```

---
## 4) 两阶段评估：Stage I（tune）与 Stage II（compare）

### 4.1 Stage I：`dmeval tune`（单场景调参 + 选 best_configs）

输入（tune 专用 YAML，不是 `config.py` 的 DMEvalConfig 格式）
- CLI：`dmeval tune --config tune.yaml`
- `tune.yaml` 的常用字段（见 `tune.py:tune_command()`）：
  - `workdir`：运行时 cwd（相对路径解析基准）
  - `pipeline_root`：流水线根目录（默认在 mpd-splines-public 的 logs 下）
  - `python`：Python 可执行文件（默认 python3）
  - `mpd_inference_script`：MPD 推理脚本路径
  - `common_inference_args`：会被转换成 `--k v` 追加到命令末尾
  - `tune.base_cfg`：单场景 base cfg（MPD inference 的 YAML）
  - `tune.samplers[]`：每个 sampler 的 name/method/grid
  - `tune.seeds`：重复实验的 seeds
  - `tune.objective.score_fn / feasible_fn`：`module:function`，动态加载
  - `tune.top_k`：每个 sampler 保存 top-k（默认 1）
  - `tune.skip_existing`：断点续跑（默认 true）
  - `tune.dry_run`：只打印命令（默认 false）

stage1_root 是怎么决定的（用于你找输出在哪里）
- `stage1_root = <pipeline_root>/stage1_tune/<scenario_name>`
- 若 `pipeline_root` 不是绝对路径，则最终会相对于 `workdir` 解析为绝对路径（见 `tune.py` 里的 `stage1_root_abs`）。

输出（目录结构）
```
<stage1_root>/
  <run_tag>/<seed>/                # 每个候选/每个 seed 的 MPD 落盘结果
    results_single_plan-*.pt
    args_inference.yaml

  _dmeval/
    generated_configs/...          # 网格展开后生成的候选 YAML（供 mpd inference 读取）
    trial_metrics.csv
    run_metrics.csv
    rank_*.csv
    tune_manifest.yaml             # 记录所有候选（可追溯）
    best_configs/
      <sampler>/
        best.yaml                  # 最佳候选的“完整配置”
        best_patch.yaml            # 只含 sampler patch + method（Stage II 复用更方便）
        topk.yaml
    best_configs/summary.yaml
```

调用图
```
用户
  │ dmeval tune --config tune.yaml
  v
cli.py:main()
  v
tune.py:tune_command(tune.yaml)
  │
  ├─ 生成候选 cfg：_grid_combinations + set_by_dotted_key
  │    └─ 写到 _dmeval/generated_configs/
  │
  ├─ 构造 JobConfig 列表（每个候选一个 run_tag）
  │
  ├─ runner.run_jobs_serial(...)  ──▶ subprocess.run(MPD inference.py ...)
  │
  ├─ collect.collect_results(adapter="mpd", results_root=<stage1_root>)
  │
  ├─ 读取 run_metrics.csv
  │
  ├─ 动态加载 objectives：
  │    ├─ score_fn = import("module:function")
  │    └─ feasible_fn = import("module:function")
  │
  └─ _select_best(...)  ──▶ 写 best_configs/<sampler>/(best.yaml, best_patch.yaml, ...)
```

最小 tune.yaml 示例（MPD）
```yaml
workdir: "."
pipeline_root: "mpd-splines-public/scripts/inference/logs/dmeval_pipeline"
python: "python3"
mpd_inference_script: "mpd-splines-public/scripts/inference/inference.py"
common_inference_args:
  n_start_goal_states: 10

tune:
  base_cfg: "abc/dmeval/configs/mpd_scene01.yaml"
  scenario_name: "scene01"
  seeds: [2, 3]
  samplers:
    - name: "ddim"
      method: "ddim"
      grid:
        sampler.num_steps: [16, 32]
        sampler.eta: [0.0, 0.1]
    - name: "ddpm"
      method: "ddpm"
      grid:
        sampler.num_steps: [32]
  objective:
    feasible_fn: "dmeval.objectives.simple:is_feasible"
    score_fn: "dmeval.objectives.simple:score"
  top_k: 3
  skip_existing: true
  dry_run: false
```

---
### 4.2 Stage II：`dmeval compare`（用 tuned 配置跨场景公平比较）

输入（compare 专用 YAML，不是 `config.py` 的 DMEvalConfig 格式）
- CLI：`dmeval compare --config compare.yaml`
- `compare.yaml` 常用字段（见 `compare.py:compare_command()`）：
  - `workdir/pipeline_root/python/mpd_inference_script/common_inference_args`（同 tune）
  - `compare.best_configs_root`：Stage I 输出的 `best_configs` 目录
  - `compare.scenarios[]`：每个元素包含：
    - `name`：场景名（用于 run_tag 与生成配置目录名）
    - `base_cfg`：该场景的 MPD base cfg YAML
  - `compare.seeds`：重复实验 seeds
  - `compare.skip_existing`：断点续跑
  - `compare.make_plots`：是否画图
  - `compare.dry_run`

stage2_root 是怎么决定的
- `stage2_root = <pipeline_root>/stage2_compare`
- 若 `pipeline_root` 不是绝对路径，则最终会相对于 `workdir` 解析为绝对路径（见 `compare.py` 里的 `stage2_root_abs`）。

输出（目录结构）
```
<stage2_root>/
  <scenario>/<sampler>/<seed>/     # Stage II 统一布局：场景/采样器/seed
    results_single_plan-*.pt
    args_inference.yaml

  _dmeval/
    generated_configs/<scenario>/<sampler>.yaml   # merge(base_cfg, best_patch) 后生成的最终配置
    trial_metrics.csv
    run_metrics.csv
    rank_*.csv
    plots/*.png   # make_plots=true 时
```

调用图
```
用户
  │ dmeval compare --config compare.yaml
  v
cli.py:main()
  v
compare.py:compare_command(compare.yaml)
  │
  ├─ 读取 Stage I 输出：
  │    best_configs_root/<sampler>/best_patch.yaml  (或兜底 best.yaml)
  │
  ├─ 对每个场景 base_cfg 做 deep_merge(base_cfg, patch)
  │    └─ 写到 stage2/_dmeval/generated_configs/<scenario>/<sampler>.yaml
  │
  ├─ runner.run_jobs_serial(...)  ──▶ subprocess.run(MPD inference.py ...)
  │
  ├─ collect.collect_results(adapter="mpd", results_root=<stage2_root>)
  │
  └─ (可选) plot.plot_from_run_metrics(...)
```

最小 compare.yaml 示例（MPD）
```yaml
workdir: "."
pipeline_root: "mpd-splines-public/scripts/inference/logs/dmeval_pipeline"
python: "python3"
mpd_inference_script: "mpd-splines-public/scripts/inference/inference.py"
common_inference_args:
  n_start_goal_states: 10

compare:
  best_configs_root: "mpd-splines-public/scripts/inference/logs/dmeval_pipeline/stage1_tune/scene01/_dmeval/best_configs"
  scenarios:
    - name: "scene01"
      base_cfg: "abc/dmeval/configs/mpd_scene01.yaml"
    - name: "scene02"
      base_cfg: "abc/dmeval/configs/mpd_scene02.yaml"
  seeds: [2, 3]
  skip_existing: true
  make_plots: true
  dry_run: false
```

---
## 5) MPD adapter 抽取哪些字段（你写 objective 时需要知道）

`adapters/mpd.py:MpdResultsAdapter.extract_row()` 最终会产出一行 `row: Dict[str, Any]`，常见字段包括：
- 元信息：
  - `scenario`, `run_tag`, `seed`, `result_file`
  - `env_id_replace`, `diffusion_sampling_method`, `planner_alg`
- 时间：
  - `t_inference_total`, `t_generator`, `t_guide`
- 核心指标（若 `.pt` 内有 metrics 且结构符合预期）：
  - `success`, `fraction_valid`, `collision_intensity`
  - `path_length_best`, `smoothness_best`, `diversity_valid`
  - 一些 EE pose error 的 mean/std 等

而 `collect.py` 聚合后（run_metrics.csv），这些数值列会变成：
- `<metric>_mean`
- `<metric>_std`

所以在 Stage I objective 里（如 `objectives/simple.py`），典型读取的是：
- `success_mean`
- `t_inference_total_mean`
- `fraction_valid_mean`
- `collision_intensity_mean`
等。
