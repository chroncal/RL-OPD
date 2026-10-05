# RL-OPD：时间线共识辅助 GRPO

探索视觉 grounding 中的时间线共识能否帮助 GRPO 改善最终定位。主方法为 **GRPO + 在线共识蒸馏**；纯 GRPO 是主对照，纯蒸馏为消融，基础模型时间线投票为不训练基线。

这是当前实验代码的可审阅快照，实验尚在运行，**不代表已证明方法有效或已完成论文实验**。代码与启动器保留原服务器路径；运行前必须准备模型、数据及对应环境。

## 方法

每条完整 rollout 在推理前缀 10%–90% 读出九个框，加上最终框。共识框为与其他九个框平均 IoU 最高的候选（不是坐标平均）；稳定度为十个框的 45 对平均 IoU。选择和门控不使用 GT。

GT 提供原始 GRPO 的分档 IoU 奖励。辅助项在学生自己的完整推理之后，只对共识答案的坐标 token 计算交叉熵：`L = L_GRPO + 0.05 * gate * L_coordinate_CE`。当前 gate 为共识有效、`0.4 <= stability < 0.9` 且共识框与最终框不同。

## 代码入口

| 内容 | 文件 |
|---|---|
| 共识选择、稳定度门控、坐标 CE | `timeline_self_distill/consensus_objectives.py` |
| 当前 rollout、九位置读框、GRPO/蒸馏接入 | `timeline_self_distill/consensus_plugin.py` |
| 三训练组 LoRA 配置及启动 | `timeline_self_distill/consensus_launch.py` |
| 工程 profile、64 步训练及最终评估 | `timeline_self_distill/consensus_workflow.py` |
| 并行调度（已有串行任务的接续入口） | `timeline_self_distill/consensus_parallel.py` |
| GT-free 投票与固定位置统计 | `timeline_self_distill/timeline_vote_statistics.py`, `fixed_position_statistics.py` |
| 跨轨迹投票对照 | `timeline_self_distill/cross_trajectory_*.py` |
| 500 条验证集比较、修好/弄坏、置信区间 | `timeline_self_distill/consensus_report.py` |
| 导出及长序列工程见证 | `consensus_export.py`, `consensus_memory_witness.py` |

保留了上述入口调用的历史辅助模块；它们不代表本轮额外实验。精确协议见 `timeline_self_distill/refine-logs/consensus_pilot64_20261005/EXPERIMENT_PLAN.md`。

## 当前探索设置

- 模型：Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`。
- FineCops-Ref 原训练 2000 问题、固定验证 500 问题；训练与验证图像完全不重叠。
- 三训练组独立从基础模型开始，单 seed 0，语言层 LoRA rank 8、alpha 16、dropout 0，视觉部分冻结。
- 每组 64 optimizer steps；每步 16 问题 × 8 rollout，共 128 条。64 步不是完整 2000 问题一轮。
- 推理 temperature 0.7、top_p 1、top_k -1，生成上限 8192，最大上下文 12288。
- 中途读框贪心；训练最终框采样；正式最终评估用贪心完整前缀读框。
- GRPO 奖励：IoU < 0.5 为 0；[0.5,0.75) 为 0.5；>= 0.75 为 1。KL 系数 0.04。
- 仅共识蒸馏组不使用 GT 训练，GRPO 项与 KL 为零。

## 运行与检查

请先阅读 [运行前提](docs/RUNTIME.md) 和 [已知限制](docs/LIMITATIONS.md)。不包含模型权重、图像、GT manifest、原始 rollout 或训练输出。

已有训练环境下的 CPU 单元检查（不启动训练）：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest timeline_self_distill.test_consensus_objectives
```

原始环境中的 profile / pilot 启动方式（会启动 GPU 工作，须先配置路径和准备缓存）：

```bash
python -m timeline_self_distill.consensus_workflow --mode profile --output /path/to/new-profile
python -m timeline_self_distill.consensus_workflow --mode pilot --profile-root /path/to/completed-profile --output /path/to/new-pilot
```

## 来源

快照来自本地实验代码提交 `431063ecc355d7e19e1006b770fda0c8a5d67177`。`SOURCE_MANIFEST.json` 保存逐文件 SHA-256；上传没有改变正在运行的实验代码或参数。
