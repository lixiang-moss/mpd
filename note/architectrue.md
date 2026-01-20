

flowchart LR
    %% ================= 左侧：输入 =================
    subgraph INPUT["输入：Hydra 配置 (YAML)"]
        I1["场景集合"]
        I2["采样器与超参数"]
        I3["搜索空间（网格 / Optuna）"]
        I4["实验开关"]
    end

    %% ================= 中央：MPD 评估框架 =================
    subgraph FRAME["MPD 评估框架（CODEX）"]
        direction LR

        A["配置与编排<br/>(Hydra)<br/><br/>实验任务<br/>(采样器 + 参数 + 场景)"]
        B["MPD 运行适配器<br/>(子进程 / CLI)<br/><br/>调用 MPD<br/>(inference.py)"]
        C["结果收集与归一化<br/><br/>原始指标<br/>(success, path_len, t_infer 等)<br/><br/>长表 CSV / Parquet"]
        D["汇总与分析<br/><br/>按任务维度分组<br/>(采样器, 参数, 场景)<br/><br/>统计量<br/>(均值, 方差, 置信区间, 比率)"]
        E["报告与可视化<br/><br/>总表<br/>(排名, 对比)<br/><br/>图表<br/>(曲线, 柱状, 雷达)"]

        A --> B --> C --> D --> E
    end

    %% ================= MPD 黑盒 =================
    subgraph MPD["MPD（黑盒）"]
        direction TB
        M1["inference.py"]
        M2["n 次试验<br/>(内部循环)"]
        M3["标准输出 / 结果文件"]
        M1 --> M2 --> M3
    end

    %% ================= 右侧：评估计划 =================
    subgraph PLAN_A["计划 A：单场景调参"]
        direction TB
        PA1["选择代表性场景"]
        PA2["遍历候选参数组合<br/>(每组 n 次试验)"]
        PA3["按成功阈值过滤（< 0.9）"]
        PA4["按 t_inference 升序排序"]
        PA5["平局处理<br/>(有效性, 质量)"]
        PA6["输出：每个采样器的最佳配置"]
        PA1 --> PA2 --> PA3 --> PA4 --> PA5 --> PA6
    end

    subgraph PLAN_B["计划 B：多采样器对比"]
        direction TB
        PB1["输入：最佳配置（来自计划 A） + 多个场景"]
        PB2["运行完整评估套件<br/>(每种组合 n 次试验)"]
        PB3["收集并汇总统一指标"]
        PB4["输出：跨采样器 / 场景的对比图"]
        PB1 --> PB2 --> PB3 --> PB4
    end

    %% ================= 连接关系 =================
    INPUT --> A
    B -.调用.-> M1
    M3 -.结果.-> C
    D --> PLAN_A
    PLAN_A --> PLAN_B
