Evaluation Pipeline Design Plan
Goal
Build a reusable evaluation pipeline that can be used to assess the effect of swapping different samplers under models , so that they can be automatically executed under the same set of rules, with automatic metric collection, automatic statistical aggregation and comparison, and finally produce data tables and plots.
In the initial stage, I will use this tool together with MPD. MPD is responsible for generating trajectories and computing metrics (the evaluation tool itself does not implement metric computation because different models may not use the same standards or data formats when computing trajectory metrics, which is not conducive to the generality principle of the evaluation tool’s design). The evaluation tool is responsible for configuration management, batch execution, data collection, aggregated statistics, visualization, and comparative conclusions.
Design Principles of the Evaluation Tool


General and extensible
The framework interacts with models via interfaces, and uses YAML configuration to support flexible extension to different task scenarios, samplers, and hyperparameters.

Decoupled from diffusion-model implementations
Do not modify the evaluated model itself; rely only on its invocation interface and output data. This makes it easy to adapt to other models that can be called from the command line and output metrics.

Configuration-driven
Use Hydra to manage YAML configurations to uniformly control scenario selection, samplers, hyperparameter combinations, and experiment sets.

Evaluation objectives are flexibly configurable
Given the diversity of trajectory evaluation metrics, the framework supports user-defined objective functions for hyperparameter tuning. Users can combine different evaluation metrics (e.g., assigning different weights to success rate, trajectory validity rate, smoothness, and end-effector trajectory error) as the optimization guidance, to flexibly handle multi-objective decision-making problems.

4. Overall Workflow

(Simplified version; many detailed handling decisions may only be determined during implementation.)

Stage 1: Interface and Configuration Integration
Verify that “using only YAML + a single call to inference.py” is sufficient to switch scenarios and sampler parameters, and that per-trial metrics can be obtained stably.



Stage 2: Single-Scenario Sampler Hyperparameter Optimization 
On one representative scenario, find the “optimal” inference hyperparameters for a given sampler.


Method: Use a tuning tool for optimization (e.g., Hydra Sweep and Optuna).

Deliverable: The sampler’s best_config (plus several alternatives).

Stage 3: Full-Scenario Experiments Under Optimal Hyperparameters 
For each sampler, run the full evaluation suite on new scenarios of different difficulty using its own optimal hyperparameter configuration; uniformly collect and aggregate results and produce comparative conclusions.


Deliverable: Cross-sampler, cross-scenario comparison tables + plots + summary.

Simplified Diagram (using MPD as an example):

Progress and Expectations
Progress

The complete project architecture has been planned, and coding can begin.
I am learning Hydra and Optuna.
I have read some papers on testing tools related to motion planning and took a quick look at the EDMP codebase of EDMP: Ensemble-of-costs-guided Diffusion for Motion Planning, to check whether this evaluation tool’s way of working with MPD via YAML or command-line instructions is suitable for evaluating other motion-planner-related models. So far, it appears feasible. If things go smoothly with MPD, I may also try it on the EDMP codebase as well.



Use MCP for automatic hyperparameter tuning.
Analyze the final plots or data and generate an analysis report.
Automatically complete the interface integration work with diffusion models.