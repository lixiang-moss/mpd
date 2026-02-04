# MPD + DMEval（Docker 内端到端测评流程：DDPM vs DPM-Solver++）

本文档面向“已经进入 MPD 的 Docker 容器”的使用场景，给出从安装 DMEval 到跑完整个评估流程（运行 → 收集 → 排名 → 可视化）的可操作步骤。

> 说明：DMEval 只负责**编排运行/提取/聚合**；轨迹指标数值由 MPD 内部计算并保存，DMEval 读取这些产物。

---
## 0. 你需要知道的两个事实（避免踩坑）

1) MPD 的推理脚本会把每次 trial 的结果保存成：
- `results_single_plan-XXX.pt`（PyTorch 序列化文件，包含 `metrics` 等）
- 同目录还有 `args_inference.yaml`（记录本次运行使用的 inference 配置）

2) DMEval 的 `mpd` 适配器做的事是：
- 扫描 `results_root/**/results_single_plan-*.pt`
- `torch.load` 加载 `.pt`
- 读取旁边的 `args_inference.yaml`
- 抽取关键字段 → 写 `trial_metrics.csv` / `run_metrics.csv` / `rank_*.csv`

---
## 1. 在 Docker 容器里准备 Python 环境

下面假设你在容器里能运行 `python3`，并且 MPD 的依赖环境已经安装好（按你的 MPD Docker 镜像为准）。

可选（如果你使用的是 MPD 推荐的 conda 环境）：
```bash
cd /workspaces/MPDLX-B-new   # 以你的容器挂载路径为准
source mpd-splines-public/set_env_variables.sh
conda activate mpd-splines-public
```

确认 Python：
```bash
python3 --version
python3 -m pip --version
```

---
## 2. 安装 DMEval（包含 MPD 适配器所需依赖）

推荐一次装齐“读取 `.pt` + 画图”：
```bash
cd mpd-splines-public
python3 -m pip install -e "dmeval[mpd,plot]"
```

验证：
```bash
dmeval --help
dmeval collect --help
```

---
## 3. 准备两份 MPD inference 配置（DDPM vs DPM-Solver++）

选一个你要评估的基础配置（示例用 Panda spheres 3D）：
```bash
BASE_CFG="mpd-splines-public/scripts/inference/cfgs/config_EnvSpheres3D-RobotPanda_00.yaml"
OUT_CFG_DIR="mpd-splines-public/scripts/inference/cfgs/generated/dmeval"
mkdir -p "${OUT_CFG_DIR}"
```

生成 DDPM 配置（只改 `diffusion_sampling_method`）：
```bash
CFG_DDPM="${OUT_CFG_DIR}/config_ddpm.yaml"
cp "${BASE_CFG}" "${CFG_DDPM}"
sed -i "s/diffusion_sampling_method: 'ddim'/diffusion_sampling_method: 'ddpm'/" "${CFG_DDPM}"
```

生成 DPM-Solver++ 配置（使用 `dpm_solver_pp` 变体；会复用 `dpm_solver:` 配置段）：
```bash
CFG_DPMSOLVERPP="${OUT_CFG_DIR}/config_dpm_solver_pp.yaml"
cp "${BASE_CFG}" "${CFG_DPMSOLVERPP}"
sed -i "s/diffusion_sampling_method: 'ddim'/diffusion_sampling_method: 'dpm_solver_pp'/" "${CFG_DPMSOLVERPP}"
```

快速检查：
```bash
rg -n "diffusion_sampling_method" "${CFG_DDPM}" "${CFG_DPMSOLVERPP}"
```

> 备注：`dpm_solver_pp` 在 MPD 里会强制使用 DPM-Solver++（见 `mpd/inference/inference.py` 对 `dpm_solver_pp` 的处理逻辑）。

---
## 4. 运行两组实验（产生 `.pt` 结果）

建议先选定：
- 同一个 `seed`
- 同一个 `selection_start_goal`（例如 validation）
- 同一个 `n_start_goal_states`（例如 50 或 100）

设置公共参数：
```bash
RESULTS_ROOT="mpd-splines-public/scripts/inference/logs/dmeval_ddpm_vs_dpmsolverpp"
SEED=2
N=50
DEVICE="cuda:0"    # 没有 GPU 就改成 cpu（但可能非常慢）
```

跑 DDPM：
```bash
python3 mpd-splines-public/scripts/inference/inference.py \
  --cfg_inference_path "${CFG_DDPM}" \
  --results_dir "${RESULTS_ROOT}/ddpm" \
  --seed "${SEED}" \
  --selection_start_goal validation \
  --n_start_goal_states "${N}" \
  --save_results_single_plan_low_mem true \
  --device "${DEVICE}"
```

跑 DPM-Solver++：
```bash
python3 mpd-splines-public/scripts/inference/inference.py \
  --cfg_inference_path "${CFG_DPMSOLVERPP}" \
  --results_dir "${RESULTS_ROOT}/dpm_solver_pp" \
  --seed "${SEED}" \
  --selection_start_goal validation \
  --n_start_goal_states "${N}" \
  --save_results_single_plan_low_mem true \
  --device "${DEVICE}"
```

运行完成后，你应该能看到类似文件结构：
```bash
find "${RESULTS_ROOT}" -maxdepth 4 -name "results_single_plan-*.pt" | head
find "${RESULTS_ROOT}" -maxdepth 4 -name "args_inference.yaml" | head
```

---
## 5. 用 DMEval（MPD 适配器）收集/聚合/排名

```bash
dmeval collect \
  --adapter mpd \
  --results_root "${RESULTS_ROOT}" \
  --out_dir "${RESULTS_ROOT}/_dmeval"
```

产物说明：
- `trial_metrics.csv`：每个 `results_single_plan-*.pt` 一行（trial 粒度）
- `run_metrics.csv`：按 `(scenario, run_tag, seed)` 聚合均值/方差（run 粒度）
- `rank_success.csv` / `rank_fraction_valid.csv` / `rank_path_length.csv` / `rank_speed.csv`：基于 run 粒度均值排序

---
## 6. 画基础对比图（可选）

```bash
dmeval plot \
  --run_metrics "${RESULTS_ROOT}/_dmeval/run_metrics.csv" \
  --out_dir "${RESULTS_ROOT}/_dmeval/plots"
```

你会得到每个 `scenario` 的：
- 成功率柱状图
- 有效率柱状图
- 推理耗时柱状图
- time-vs-success 散点图

---
## 7. 关于 `.pt` 是什么，以及为什么用它

`.pt` 是 PyTorch 生态里常见的“序列化结果文件”扩展名，本质上是：
- `torch.save(python_object, path)` 写入
- `torch.load(path)` 读出

优点（对 MPD 这类项目）：
- 能直接保存/恢复 **Python 对象 + Tensor**（比如大量轨迹张量、指标对象）
- 不需要你手写 JSON/CSV 的字段映射（但会有可移植性/安全性注意事项）

注意事项（重要）：
- `.pt` 底层使用 pickle 体系，**不要加载不可信来源的 `.pt`**（可能有安全风险）。
- 为了节省空间，MPD 提供了 `--save_results_single_plan_low_mem true`，只保存评估所需的关键字段。

---
## 8. “低内存保存”该放在 DMEval 还是 MPD？

结论建议：
- **更适合在 MPD（规划器侧）实现**：因为只有规划器最清楚“哪些中间变量可以不存、哪些必须存”，并且不同规划器的“中间过程”完全不同。
- DMEval 更适合做两件事：
  1) **兼容读取**（无论你存 full 还是 low-mem，只要产物里有 `metrics`/时间/必要字段都能收集）
  2) **在 `run` 阶段把参数透传给规划器**（例如在 job 里固定 `--save_results_single_plan_low_mem true`）

如果未来你希望“通用化”这件事，建议把它抽象成：
- 规划器输出契约的一个选项：`artifact_profile: full|metrics_only|metrics+best_traj`
而不是让 DMEval 去“事后裁剪”规划器的内部对象。

---
## 9. 可选：用 dmeval tune/compare 跑两阶段流程（更贴近 proposal01）

如果你希望严格按两阶段范式执行：
- Stage I：单场景网格调参 → 输出每个 sampler 的 `best_configs`
- Stage II：加载 best_configs，在多个场景重新跑对比

可以直接参考并修改以下示例配置：
- `mpd-splines-public/dmeval/configs/example_mpd_tune.yaml`
- `mpd-splines-public/dmeval/configs/example_mpd_compare.yaml`

说明：这两份配置使用 Hydra `defaults` 组合配置（`planner/`、`scenario/`、`objective/`），并支持用 `--override key=value` 临时覆盖参数（仍然串行执行）。

执行：
```bash
dmeval tune --config mpd-splines-public/dmeval/configs/example_mpd_tune.yaml
dmeval compare --config mpd-splines-public/dmeval/configs/example_mpd_compare.yaml
```

示例：覆盖场景与试验次数：
```bash
dmeval tune --config mpd-splines-public/dmeval/configs/example_mpd_tune.yaml \
  --override scenario=warehouse \
  --override common_inference_args.n_start_goal_states=20
```
