# MiniMind 从零到复现学习路线（面向已实现过 Transformer 的你）

> 目标：你已经读过并手写过 Transformer，这份路线将帮助你把 MiniMind 当作「完整 LLM 工程课」来系统学习，并在最后独立复现一个可对话的小模型。

---

## 0. 先定学习策略：两条线并行

- **实现线（How）**：先跑通，再拆解代码细节。
- **原理论（Why）**：每读一个模块，都回到论文/公式解释其必要性。

建议按「**先 Dense，后 MoE；先 SFT，后 RL**」推进，不要一开始把所有训练脚本都混在一起。

---

## 1. 项目全景（先看 2 小时）

### 1.1 你要知道这个仓库覆盖了什么

MiniMind 不只是模型结构，还覆盖了 Tokenizer、预训练、SFT、LoRA、DPO、PPO/GRPO/CISPO、Tool Use、Agentic RL、蒸馏与推理部署。你可以把它当作一个「完整训练链路样例」。

### 1.2 建议先看哪些文件

1. `README.md`：只看「快速开始」「模型训练」「模型结构」章节。
2. `model/model_minimind.py`：模型主干（最核心）。
3. `dataset/lm_dataset.py`：数据如何变成训练张量。
4. `trainer/train_pretrain.py` 与 `trainer/train_full_sft.py`：两段主线训练流程。
5. `trainer/trainer_utils.py`：初始化、学习率、checkpoint、恢复训练。

---

## 2. 第一阶段（第 1 周）：跑通主线最小闭环

> 目标：不纠结细节，先获得「可运行、可观察、可改动」的完整体验。

### Day 1：环境 + 推理跑通

- 安装依赖。
- 下载官方模型，跑 `eval_llm.py`，确保推理可用。
- 额外做一件事：开 `web_demo.py` 或 API 服务脚本，感受 chat_template / tool call / think 输出的行为差异。

### Day 2-3：训练最小实验（建议 mini 数据）

- 下载 `pretrain_t2t_mini.jsonl`、`sft_t2t_mini.jsonl`。
- 跑一次 `train_pretrain.py`（少量 step 即可）。
- 接着跑 `train_full_sft.py`，把预训练权重接上。
- 最后用 `eval_llm.py --weight full_sft` 验证输出质量变化。

### Day 4-5：断点续训 + 多卡入口理解

- 故意中断训练，再用 `--from_resume 1` 恢复。
- 理解 DDP 入口（`torchrun --nproc_per_node N ...`）和日志逻辑。
- 你只需确认：checkpoint 里到底保存了哪些状态、跨卡数恢复如何做 step 修正。

### Day 6-7：做第一份学习报告（非常重要）

写一页笔记回答：

1. MiniMind 从输入文本到 loss 的路径是什么？
2. pretrain 与 SFT 在数据构造上的差异是什么？
3. 你当前设备最可能的瓶颈是显存、算力、I/O 还是数据处理？

---

## 3. 第二阶段（第 2 周）：逐模块精读（模型 + 数据 + 训练）

> 目标：把「会跑」升级成「我能解释每一行关键代码」。

### 3.1 模型模块精读顺序（`model/model_minimind.py`）

按这个顺序读：

1. `MiniMindConfig`：先搞清参数语义（`hidden_size`、`num_hidden_layers`、`num_attention_heads`、`num_key_value_heads`、`rope_theta`）。
2. `RMSNorm`、RoPE (`precompute_freqs_cis` / `apply_rotary_pos_emb`)。
3. `Attention`：重点看 GQA（q_heads ≠ kv_heads）、cache 拼接、flash attention 回退逻辑。
4. `FeedForward`（SwiGLU）与 `MOEFeedForward`（top-k route + aux loss）。
5. `MiniMindBlock` / `MiniMindModel`：残差流、层堆叠、位置编码缓存。
6. `MiniMindForCausalLM.forward()`：loss 构造与 label shift。
7. `generate()`：采样参数、KV cache 增长、终止条件。

### 3.2 数据模块精读（`dataset/lm_dataset.py`）

- `PretrainDataset`：纯文本 next-token，pad 位 label 置 `-100`。
- `SFTDataset`：chat_template + 只对 assistant 区间计算 loss。
- `DPODataset`：chosen/rejected 双样本 + mask。
- `RLAIFDataset` / `AgentRLDataset`：prompt 采样与工具轨迹输入。

**练习建议**：打印 5 条样本，手工检查 token 与 label 对齐是否符合预期。

### 3.3 训练框架精读（`trainer/train_pretrain.py` + `trainer/trainer_utils.py`）

重点抓 4 件事：

1. 学习率曲线 `get_lr` 是怎么定义的。
2. 梯度累积 + AMP + grad clip 顺序。
3. checkpoint 保存了模型、优化器、step、可视化 run id。
4. `init_model` 如何根据 `from_weight` 衔接阶段训练。

---

## 4. 第三阶段（第 3 周）：你自己从头复现一遍（核心挑战）

> 目标：不是“读懂仓库”，而是“你能重建仓库的主线能力”。

### 4.1 复现任务切分

你可以新建一个 `my_minimind_clone/`，只做以下最小集合：

- `config.py`：最小配置类。
- `model.py`：RMSNorm + RoPE + Attention + FFN + Decoder block + CausalLM head。
- `dataset.py`：PretrainDataset + SFTDataset（先不做 DPO/RL）。
- `train_pretrain.py`：单卡训练。
- `train_sft.py`：接 pretrain 权重。
- `infer.py`：greedy + top-p 采样。

### 4.2 验收标准（务必量化）

1. 能在 mini 数据上稳定降 loss。
2. SFT 后能完成基础问答。
3. 你能说清楚每个张量 shape 的变化。
4. 你能独立排查 2 个以上训练 bug（例如 label 错位、mask 错误、cache shape 不匹配）。

---

## 5. 第四阶段（第 4 周）：扩展专题（按兴趣选）

### 5.1 MoE 专题

- 打开 `use_moe=1`，比较 Dense vs MoE 的训练速度、loss 曲线、显存。
- 重点观察 `aux_loss` 是否有效避免 expert 塌缩。

### 5.2 对齐训练专题（DPO / PPO / GRPO / CISPO）

- 先 DPO，再 PPO/GRPO。
- 不要先纠结 SOTA，先确认「奖励建模与策略更新的数据流」完整跑通。

### 5.3 Tool Use / Agentic RL 专题

- 读 `train_agent.py` 与 `rollout_engine.py`。
- 做一个最小工具：计算器或天气查询（mock 也行），观察多轮 tool call 轨迹质量。

---

## 6. 建议你的每日学习模板（90~120 分钟）

1. **20 分钟**：读 100~200 行源码，画流程图。
2. **40 分钟**：跑一个最小实验（即使只跑 100 step）。
3. **20 分钟**：写当天笔记（输入输出、shape、loss、异常）。
4. **10~20 分钟**：第二天的问题清单。

> 关键原则：每天都要「读代码 + 跑实验 + 写结论」，三者缺一不可。

---

## 7. 一份给你的「高性价比学习清单」

### 必做（强烈建议）

- 手动画出 MiniMind forward 图（从 token 到 logits 到 loss）。
- 自己实现一版 `SFTDataset.generate_labels`，并和仓库输出逐 token 对比。
- 手改一个超参数（如 `num_key_value_heads` 或 `max_seq_len`），观察训练速度与效果。
- 至少完整跑通一次：pretrain → SFT → eval。

### 选做（进阶）

- 把某个 trainer 重构成你喜欢的风格（如 lightning 风格，但保留原始逻辑）。
- 将你复现版导出到 transformers 兼容格式（对照 `scripts/convert_model.py`）。
- 增加一个你自己的评测脚本（例如固定 50 条中文问答，比较版本差异）。

---

## 8. 常见学习误区（提前避免）

1. **误区：只看 README，不跑代码。**
   - 解决：任何新概念都配一个最小实验。

2. **误区：一上来就做 RL。**
   - 解决：先把 pretrain + SFT 跑稳，RL 才不会变成“盲调参数”。

3. **误区：只盯模型，不看数据处理。**
   - 解决：SFT 质量高度依赖模板与 label mask，数据错误会让你误判模型结构。

4. **误区：过早追求大规模训练。**
   - 解决：先在 mini 数据验证正确性，再扩规模。

---

## 9. 我给你的起步执行计划（今天就能开始）

1. 读 `README.md` 的训练章节，记录你本机可用设备与预算。
2. 跑 `eval_llm.py`，确认推理链路。
3. 下载 mini 数据集，跑 pretrain 100~500 step。
4. 接着跑 SFT 100~500 step。
5. 写下你遇到的第一个 bug（或性能瓶颈），明天开始针对性深挖。

如果你愿意，我下一步可以直接给你做一份**「按你当前机器配置（GPU/显存/每天可学时间）」定制的 14 天打卡表**，每天明确到：读哪些行代码、跑哪个命令、验证什么现象。
