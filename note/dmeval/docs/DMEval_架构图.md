# DMEval 一页架构图

> 目标：用“**两阶段评估范式**（先调参、后跨场景对比）”在统一规则下评估不同扩散采样器。
>
> 关键原则：**不改 MPD**；指标由 MPD 内部计算；DMEval 只负责“编排运行/提取/聚合/对比/可视化”。
>
> 当前版本：**串行执行**；配置管理使用 **Hydra（compose API）**。

---
## 1) 总体流程图（Stage I → Stage II）

```
┌───────────────────────────────┐
│  输入：Hydra YAML 配置（配置驱动） │
│  - defaults 组合场景/采样器/目标函数│
│  - --override 临时覆盖参数         │
│  - 场景 base_cfg（MPD YAML）   │
│  - sampler + 超参搜索空间       │
│  - seeds / n_start_goal_states │
│  - 目标函数 objective           │
└───────────────┬───────────────┘
                │
                v
┌──────────────────────────────────────────────────────────────┐
│ Stage I：dmeval tune（单场景调参 / 网格搜索 / 串行）              │
│ 1) 生成候选 configs（很多份 YAML）                             │
│ 2) 串行调用 MPD inference.py 跑每个候选配置                     │
│ 3) 收集所有 trial 结果 -> trial_metrics.csv                    │
│ 4) 聚合成 run_metrics.csv（mean/std）                          │
│ 5) objective 选每个 sampler 的 best_config                     │
│ 6) 输出 best_configs/<sampler>/best.yaml (+ best_patch.yaml)  │
└───────────────┬──────────────────────────────────────────────┘
                │ best_configs（每个 sampler 一套最优超参）
                v
┌──────────────────────────────────────────────────────────────┐
│ Stage II：dmeval compare（跨场景公平比较 / 串行）                │
│ 1) 读取 best_patch.yaml                                       │
│ 2) 对每个新场景 base_cfg 生成“场景专用最终配置”                   │
│ 3) 串行调用 MPD inference.py 复跑（不同 sampler / 不同场景）     │
│ 4) 收集 trial -> 聚合 run -> 排名/图表                          │
└───────────────┬──────────────────────────────────────────────┘
                │
                v
┌───────────────────────────────┐
│ 输出：对比数据表与图           │
│ - trial_metrics.csv            │
│ - run_metrics.csv              │
│ - rank_*.csv                   │
│ - plots/*.png（可选）          │
└───────────────────────────────┘
```

---
## 2) 模块架构图（DMEval 内部组件）

```
                     ┌──────────────────────────┐
                     │          CLI             │
                     │  dmeval run/collect/...  │
                     └───────────┬──────────────┘
                                 │
          ┌──────────────────────┼───────────────────────┐
          │                      │                       │
          v                      v                       v
┌───────────────────┐  ┌───────────────────┐   ┌───────────────────┐
│ Runner (串行执行)  │  │ Collector (收集)  │   │ Plot (可视化)      │
│ - subprocess.run   │  │ - 扫描结果树      │   │ - 读 run_metrics   │
│ - stdout/stderr落盘│  │ - Adapter提取字段 │   │ - 生成基础图        │
└─────────┬─────────┘  │ - 输出 CSV+rank    │   └───────────────────┘
          │            └─────────┬─────────┘
          │                      │
          v                      v
┌───────────────────┐  ┌────────────────────────────────────┐
│ MPD inference.py   │  │ Adapter（结果契约层）                │
│ （外部子进程）     │  │ - MPD: 读 .pt + args_inference.yaml   │
│ 输出：             │  │ - 未来可扩展：EDMP/...                │
│ - results_single_* │  └────────────────────────────────────┘
│ - args_inference   │
└───────────────────┘

Tune/Compare 是“流程编排器”：
- tune = Runner + Collector + Objective + BestConfigsWriter
- compare = BestConfigsReader + Runner + Collector (+ Plot)
```

---
## 3) 关键数据契约（MPD 与 DMEval 的交互点）

### 3.1 MPD 侧产物（DMEval 不依赖 sweep 文件作为数据来源）

```
<results_dir>/<seed>/
  args_inference.yaml                 # 本次 run 的配置记录
  results_single_plan-000.pt          # trial 0（包含 metrics）
  results_single_plan-001.pt          # trial 1
  ...
```

说明：
- 一次运行 inference.py 可以内部跑 n 次（n_start_goal_states），所以会生成 n 个 `.pt`。
- `.pt` 里包含 MPD 内部算好的 `metrics` 与计时字段（DMEval 只提取，不重算）。

### 3.2 DMEval 侧表格（先收集聚合，再分析/比较）

- `trial_metrics.csv`：每个 `results_single_plan-XXX.pt` 一行（细粒度）
- `run_metrics.csv`：按 `(scenario, run_tag, seed)` 聚合 mean/std（用于公平比较）

---
## 4) 文件“定位速查”

```
dmeval/
  src/dmeval/
    cli.py                 # 命令行入口与子命令分发
    runner.py              # 串行执行器（subprocess.run）
    adapters/
      base.py              # Adapter 接口协议 + TrialArtifact
      mpd.py               # MPD 结果提取（读 .pt + args_inference.yaml）
    collect.py             # 收集/聚合/排名：trial_metrics/run_metrics/rank_*.csv
    objectives/simple.py   # 占位目标函数（后续替换成论文版 objective）
    tune.py                # Stage I：调参（网格搜索）+ 输出 best_configs
    compare.py             # Stage II：读 best_configs 跨场景重跑对比
    plot.py                # 基础可视化（可选）
```

---
## 5) 当前限制与下一步

- 当前限制：Stage I 超参搜索 = **串行网格搜索**；目标函数 = 占位版；统计分析较基础（mean/std + 排名 + 基础图）。
- 下一步增强（论文级）：接入 Optuna/Hydra sweep（更高效搜索）、固化阈值+优化目标、补齐更完整的统计检验与图表清单。
