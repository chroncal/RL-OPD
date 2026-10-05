# 运行前提

本快照保留原始实验实现，未做服务器路径迁移。复现前需检查以下前提。

## 原始环境记录

| 环境 | torch | transformers | vLLM | ms-swift | PEFT |
|---|---|---|---|---|---|
| 训练与 rollout 服务 | 2.10.0 | 5.16.1 | 0.18.1 | 4.6.0.dev0 | 0.20.0 |
| 最终推理 | 2.10.0 | 4.57.6 | 0.18.1 | — | — |

这些是已安装环境的版本记录，不是通用安装保证；开发版 Swift/Transformers 与原环境修复也可能影响兼容性。代码尚未在干净机器上验证。

## 路径和输入

- 模型：`/root/rl_opd_v1_runtime/models/Qwen3.5-2B`。
- 训练环境：`/root/timeline_grpo_train_env`；推理环境：`/root/best_moment_vllm_env`；CPU 分析环境：`/root/rl_opd_curve_plot_env`。相关常量在 `e3_launch.py`、`timeline_workflow.py` 等模块中。
- 在包目录准备 `data/train2000.jsonl` 和 `data/val500.jsonl`。每行至少包含 `sample_id`、`image_path`、`expression`、`width`、`height`、`gt_xyxy`（像素 XYXY）。训练/验证图像不得重叠。原 manifest 未上传，重新选取样本不能声称精确复现原成绩。
- `prepare.py` 是历史数据准备模块，调用外部 `rl_opd_v1.build_manifests` 并需要原注释和图像索引；该外部数据准备器不在本快照内。建议显式准备 manifest，勿把该脚本视为完整下载器。
- 基础投票评估依赖 `results/e0_formal/raw/<sample_id>.json` 的原缓存。固定位置统计依赖原训练读框缓存。原始数据不在 GitHub，因此这些入口不是下载后即可运行。
- LoRA 目标元数据已包含于 `refine-logs/timeline_reformulation_20261004/LORA_META_TARGETS.json`。
- 训练 checkpoint 使用本地 `/root/timeline_e3_checkpoints`，最终复制到输出目录；原设计避开对象挂载的序列化限制。

`consensus_parallel.py` 是原运行中的并行接续工具，需要已有 registered/launch/manifest 文件，不是独立的全新任务启动器。不要把它指向正在运行的旧输出目录重新执行。
