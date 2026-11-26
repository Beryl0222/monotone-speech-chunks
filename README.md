# 单调分块注意力的等待概率与字幕提交

这是面向monotone-chunks的 Python 数值项目资料。数据全部由本地确定种子生成，不含真实业务或用户记录。

`data/materials.json` 保存张量、身份、参数与容量配置，`data/events.json` 保存有先后关系的操作输入。数学语义、状态含义与数值范围见 [数据约定](docs/data-contract.md)。事件记录不包含计算结果，不能视作已经执行的回执。

使用 Python 3.11，依赖版本记录在 `requirements.txt`。可在独立环境中运行：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python inspect_materials.py
.venv/bin/python -m unittest discover -s tests -v
```

读取入口会核对资料结构、数值有限性及引用，打印资料指纹和操作数；这些检查只说明输入可读取，不宣称算子、训练或业务流程已经执行。容量配置属于可生成输入的规格，不是测得的性能。实际算子运行入口及用法由其实现版本提供。

较大输入可用 `.venv/bin/python make_capacity.py --output /tmp/product-capacity` 生成到一个不存在的目录。它只创建输入数组和元数据，不运行算子；`arrays.npz` 的键、形状和类型记录在生成目录的 `metadata.json`。同一种子得到相同数组内容。

## 实现入口

核心包是 `monotone_chunks/`，只依赖 NumPy：

- `boundary_distribution(e, constraints, ...)`：项目自实现的单调路径 DP，返回 `alpha`、逐词 `wait`、`total_waiting_mass` 和约束路径总质量 `path_mass`。不枚举路径，也不把 argmax 分数当作软分布。
- `window_contexts(u, V, chunk_width, valid)` / `context_forward_backward(...)`：按 `[max(0,j-chunk_width+1), j]` 做块内 softmax、上下文和窗口归属梯度。
- `train_forward_backward(...)`：`loss = -log(path_mass) + 调用者上下文损失`，返回 `grad_e/grad_u/grad_V/grad_Ks/grad_Kc/grad_Q/grad_b_select`；`context_upstream` 是调用者给出的上下文伴随余量。
- `StreamingReader`：持续 `audio_arrive/push_chunk`、`online_propose/read`、`commit`、`withdraw_uncommitted`、`revise_annotation`。正式游标只由成功提交移动；临时提议标记为 `status="provisional"`、`committable=False`，撤回后下一词仍从最后正式边界读取。
- `replay_events(...)` / `python -m monotone_chunks ...`：按 `data/events.json` 的顺序重放。根目录的 `solution.py`、`core.py`、`streaming.py`、`training.py` 等是常见导入名的兼容转发层。

一个最小在线片段：

```python
from monotone_chunks import StreamingReader

reader = StreamingReader.from_materials(materials)
reader.audio_arrive(start=0, stop=8, confirmed_stop=4)
proposal = reader.online_propose("w0")
if proposal.committable:
    reader.commit("w0")
else:
    print(proposal.status, proposal.reasons, proposal.acoustic_range)
```

训练软分布和在线阈值不是同一套计算：软分布是所有非递减合法路径的概率归约，在线决定是从最后正式边界扫描第一个 `p >= tau` 的已确认边界。`reader.soft_relationship(word)` 或 `explain_soft_and_online(reader, word)` 会同时返回二者、该边界的 `alpha`、等待质量和提交/等待原因。

## 状态与结束约定

- 已收到但块末仍未选择的质量保留在 `wait`；只有 `force_end=True` / `seal_utterance(...)` 后的真句末才把剩余质量按公开约定交到最后有效帧。
- `valid=False` 是张量补齐/无效位置，不会成为边界或窗口上下文；批处理辅助 `train_tensor_batch` 逐句独立构造窗口，不共享相邻批条目。
- 多条人工时窗逐记录相乘。`outside_factor=0` 是硬排除，正数是软置信权重。
- `assess_sentence(...)` 返回 `materials_incomplete`、`constraints_infeasible/conflict`、`insufficient_allowed_path_mass` 等分类、`allowed_path_count`、`path_mass_denominator` 和所需覆盖位置。只有完整覆盖且几何/权重允许的句子进入 `train_batch` 的句权分母 `eligible_weight_denominator`。

运行数值、梯度和事件状态检查：

```sh
python -m unittest discover -s tests -v
python -m monotone_chunks --materials data/materials.json --events data/events.json
```

容量形状上限和重放块预算记录在 `monotone_chunks.workspace.LIMITS`（4096 帧、128 词、d=16、`chunk_width<=64`、最多 64 遍声学块重放）。读取状态本身只保留 O(T) 的当前归约数组和紧凑提交记录，不保存指数路径或所有解码步的分数历史。
