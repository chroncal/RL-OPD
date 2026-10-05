# 时间线共识自蒸馏：64 步单 seed 探索对照

日期：2026-10-05。用户已确认 LoRA、每组64步。新请求授权训练，取代此前统计阶段不训练的限制。

## 问题与判别
检验 GT-free 时间线 IoU 共识是否能蒸馏到最终单次定位，及其与 GRPO 是否互补。比较四组：基础模型直接时间线投票（缓存验证读框，不训练）、纯 GRPO、GRPO+共识蒸馏、仅共识蒸馏（训练不使用 GT）。三训练组从同一官方基础模型各自重新开始，不复用任何旧70样本适配器。

主指标为固定原始 val500 的最终框 IoU>=0.5准确率；逐条统计基础最终错→当前对、基础最终对→当前错。最后checkpoint64一次评估，无中间验证选择。联合优于GRPO才支持互补；纯蒸馏优于基础才支持老师独立有用；无提升或置信区间跨零保留为负结果/不确定。探索单seed不能提供论文级稳健性结论。

## 模型、输入输出、数据
Qwen3.5-2B，官方缓存 /root/rl_opd_v1_runtime/models/Qwen3.5-2B，revision15852e8c16360a2fea060d615a32b45270f8a8fc。输入图像和指代表达，输出自然推理及0..1000整数XYXY bbox_2d。使用原始train2000全部问题，不额外筛选/调整配比；val500与训练图像ID无交集（1922/429张图）。不读取sealedtest或官方testGT。

同语言层LoRA：186线性模块、rank8/alpha16/dropout0，视觉参数冻结；lr1e-4，原生AdamW，seed/data_seed0。每组64优化步，每步16问题×8rollout=128；共1024问题曝光、8192rollout，预算并非完整2000问题一轮。三组同样的问题顺序、生成参数和更新步数。另有工程检查每组2步，共6步，权重不复用到正式运行。

## 生成、共识与损失
Swift原生on-policy GRPO，num_iterations1，temperature.7/top_p1/top_k-1，推理上限8192，总长度上限12288。原生完整推理后各组统一从其原始token前缀重新以temperature.7/128token/整数bbox正则生成最终动作（保持现有格式修复协议）。所有rollout，包括达到推理上限者，保留。

联合/纯蒸馏的每条当前rollout都在10%..90%九个位置贪心读框，每次保留同一图像和精确原始token前缀，强制闭合think再读框。100%候选是本rollout实际采样最终动作。投票使用10框的精确IoU medoid，与另外9框平均IoU最高，平手取最早位置；稳定度为45对平均IoU；无效框的pair IoU为0。选择和门控绝不使用GT。

默认门控：0.4<=稳定度<0.9 且共识框不同于最终框且老师有效。相同答案的CE并不自动为0，因此显式跳过稳定/不变框以实现用户的“不起作用”意图。用户已被询问替代门控；无另行回复采用此默认。

辅助损失仅bbox坐标数字token平均CE，条件是学生自身完整推理前缀；不监督推理token，不使用早期推理作学生前缀。门控闭合轨迹损失0，batch分母始终128（不按活跃样本重新归一化）；lambda0.05，两蒸馏组相同。

GRPO奖励默认三档：IoU<.5为0，.5<=IoU<.75为.5，>=.75为1。历史实现实际使用连续IoU，此为事先登记的新奖励，两RL组同样使用。用户被询问替代连续奖励；无另行回复采用三档。原生组内标准化优势和KL beta.04。
纯蒸馏训练manifest删除solution及GT相关列，无IoU奖励、无reference KL（beta0），dummy reward0只用原生rollout排程，assert原生GRPO项严格0，只更新辅助CE。GT仅在固定val500评分阶段使用。
格式约束后的动作仍以全词表原生GRPO likelihood surrogate训练，未实现正则归一化策略概率；此限制三组一致，不能主张严格约束策略GRPO。纯GRPO不额外读九框。

## 评估和成本
保持既定基础评估格式：自然推理temperature.7，最终框贪心，官方原始tokenizer/processor、同一GT解析。val500四条原始caps仍保留。基础缓存最终274/500=54.8%，时间线投票290/500=58.0%，修好25/弄坏9。68.6%来自train1994投票，另列为训练参考，不冒充验证基线。

记录实际训练/采样GPU时间、各组活跃门控占比、生成token、caps、无效框、真实参数变化、原始采样与全部读出。九次读框不保证比完整推理便宜；不以输出token代替包含图像/前缀prefill的实际成本。按相同更新步数比较方法效果，并同时报告不等计算成本。

## 执行
仅GPU0推理/GPU1训练；不使用GPU2..7。复用温启动环境，不重装软件。每组结束清理owned processes，LoRA在本地合并保存后拷至实验目录；工程阶段真实重新加载每组导出模型做合成无GT推理证实格式与权重可用。

预部署：CPU科学检查、fresh code review CLEAR、提交源码。工程命令（从父目录执行）：
```
/root/rl_opd_curve_plot_env/bin/python -m timeline_self_distill.consensus_workflow --mode profile --output timeline_self_distill/results/consensus_profile6_20261005
```
全部工程检查通过后，重新从base启动正式命令：
```
/root/rl_opd_curve_plot_env/bin/python -m timeline_self_distill.consensus_workflow --mode pilot --profile-root timeline_self_distill/results/consensus_profile6_20261005 --output timeline_self_distill/results/consensus_pilot64_20261005
```
新失败输出保留；修复不覆盖原结果。记录提交、完整argv、环境、seed、数据模型路径；训练之后才读取评估GT做比较，最后只报告已实际完成的数字。

### 工程等待时限修正（正式训练前，2026-10-05）
2-step工程检查的联合组实测311秒/更新，推算64步约5.5小时，原实现6小时等待上限不足以容纳普通推理长度波动。将每组进程等待上限改为12小时、三组训练协调器上限36小时，接近两倍实测预估后清理。此为失败判定的工程时限，更新预算仍严格64，科学协议/数据/损失/评价不变；没有查看新验证结果。Profile仍使用原2步和7200秒协调时限，源码568889b记录保留。

### 真实OOM修复登记（2026-10-05，未进行新val评估）
原正式输出results/consensus_pilot64_20261005在GRPO第4次更新的反向OOM，完成3次更新后整体协调器退出；日志/所有128条失败window轨迹/清理记录保留。无checkpoint保存，未开始联合或纯SD正式组。OOM申请7.61GiB，对应8220×248320的FP32张量；此前8221-token更新成功，失败时有7.74GiB未使用reserve，优先检验allocator碎片修复。
统一两种Torch allocator变量为expandable_segments:True，记录effective环境；不改8192上限、样本、奖励、CE、门控、LoRA或64次有效更新。
正式恢复前，用原失败window128张量和原始精确prefix进行无生成、无优化更新、无验证集的真实GPU反向检查；同时覆盖GRPO和GRPO+坐标CE图，重复短长交替。这里自己的finalbbox仅用作辅助图的shape代表，不作教师效用评价。失败3步仅为损失的工程开销，原目录不能作为完成科学实验；完整有效对照仍从base各64步、单seed、最后val500。

### 导出冻结参数修复（同次工程修复，无新验证结果）
独立审查发现旧导出直接BF16加载会舍入基础模型36个原生FP32线性注意力参数。改用训练的Swift loader，合并前后逐值断言这36个冻结参数保持FP32且一致；另做保存后原始safetensors逐值检查和合成图像加载。旧工程导出仅证明格式可加载，不证明冻结参数精度；不复用旧导出做正式评估。


### 并行调度修正（2026-10-05，按用户最新要求）
原GPU0/1限制由本次用户明确要求并行运行取代：保留进行中的GRPO（0 rollout/1 trainer），联合组用2/3，纯蒸馏组用4/5；6/7不使用。各组科学配置、原始manifest、seed0、64次更新和最终val500均不变。独立HTTP端口8948/8958/8968，权重同步NCCL端口51216/50958/50968，MASTER端口9048/9058/9068。复用已验证环境，不重装。
新协调器consensus_parallel接管相同结果目录；后两组输出目录由各自进程独占。已加载的原串行launcher在GRPO64完成、导出和清理后尝试重复创建joint目录，预期退出1（FileExistsError），新协调器验证这一交接及全部三组64步后才评估。该预期交接不标作科学训练失败，原始日志保留。GRPO已有更新不重启，后两组从base开始，首次实际有限更新确认新设备执行。逐组仍按两张L20占用时间记录成本，并行墙钟时间另记。
