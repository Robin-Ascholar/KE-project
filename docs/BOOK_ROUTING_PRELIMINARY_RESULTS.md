# Book-aware Soft Routing 实验总结（2026-09-13）

## 实验设置

- 数据：七册问题按书册分层随机抽取 20% 开发集，共 202 题
- 随机种子：`20260913`
- 检索：原始 LinearRAG，Top-20
- 路由：各册 Top-3 passage 相似度均值，Softmax temperature `0.05`
- 指标：参考答案规范化后是否出现在检索 passage 中（Contain-Acc）
- 本轮不调用回答模型，也不使用 LLM 裁判

`oracle` 使用问题集所属书号，只表示理论上限；其余预测路由变体仅根据问题文本与 passage embedding 预测书册。

## Strength = 0.50

| 变体 | Contain@5 | Contain@10 | Contain@20 | MRR@20 | 目标书册占比@20 |
|---|---:|---:|---:|---:|---:|
| Baseline | 26.73% | 35.15% | 43.56% | 0.1803 | 44.73% |
| Oracle soft routing | 28.71% | 37.62% | 44.55% | 0.2040 | 89.68% |
| Predicted soft routing | 26.73% | 35.15% | 40.59% | 0.1870 | 56.88% |
| Confidence-aware routing | 27.23% | 34.65% | 42.08% | 0.1813 | 52.33% |

预测路由 Top-1 书册准确率为 61.88%，平均熵置信度为 0.2857。相对 Baseline：

- Oracle：新增 7 个 Contain@20 命中，丢失 5 个，净增 2 个。
- Predicted：新增 1 个，丢失 7 个，净减 6 个。
- Confidence-aware：新增 1 个，丢失 4 个，净减 3 个。

## Strength = 0.25，Confidence-aware

| 变体 | Contain@5 | Contain@10 | Contain@20 | MRR@20 | 目标书册占比@20 |
|---|---:|---:|---:|---:|---:|
| Baseline | 26.73% | 35.15% | 43.56% | 0.1805 | 44.73% |
| Confidence-aware routing | 28.22% | 34.65% | 43.07% | 0.1791 | 50.25% |

较弱路由将 Contain@5 提高 1.49 个百分点，但 Contain@20 下降 0.49 个百分点；相对 Baseline 新增 1 个 Top-20 命中、丢失 2 个。

## 最后一轮：Boost-only + Confidence-aware

为避免压低其他六册中的有效证据，最后测试了更保守的 Boost-only 方案。它只提升概率高于均匀先验的书册，其他书册权重保持 1，不做降权；其余设置与 Strength = 0.50 实验完全一致。

| 变体 | Contain@5 | Contain@10 | Contain@20 | MRR@20 | 目标书册占比@20 |
|---|---:|---:|---:|---:|---:|
| Baseline | 26.73% | 35.15% | 43.56% | 0.1805 | 44.63% |
| Boost-only + confidence | 27.72% | 35.15% | 42.08% | 0.1790 | 51.49% |

该方案在 Top-5 上提高 0.99 个百分点，但 Top-10 不变，Top-20 下降 1.49 个百分点，MRR 也略降。配对结果为新增 1 个、丢失 4 个 Top-20 命中，净减 3 个；202 道题中有 192 道的 Top-20 排名发生变化。因此，这不是一个稳定优于 Baseline 的结果。

具体得失也符合这一判断：

- 唯一新增命中是 Book 7 的 Resurrection Stone 问题，相关 passage 被提升到第 19、20 位。
- Book 4 的 Priori Incantatem 问题被错误路由到 Book 7，原本第 5 位的命中跌出 Top-20。
- Book 7 的 Fenrir Greyback 问题被错误路由到 Book 3，原本第 7 位的命中跌出 Top-20。
- Book 7 的 Rufus Scrimgeour 问题虽然正确预测为 Book 7，但 Baseline 的命中来自 Book 6；提升同册候选后，这条跨册证据仍被挤出 Top-20。Boost-only 不降低其他书册的绝对权重，却仍会通过相对重排挤出跨册证据。

## 结论与原因

当前实现不能证明 Book-aware Soft Routing 改善了检索，因此停止继续调参，也不进入 1,000 题回答生成。它可以作为一次有数据支撑的负面实验保留，但暂时不应写成已经有效的系统创新。主要原因如下：

1. **理论上限有限。** `oracle` 的 Contain@20 也只比 Baseline 高 0.99 个百分点，说明单独加入书册先验的提升空间很小。
2. **书册预测不够可靠。** 当前 Top-1 准确率为 61.88%；错误书册先验会直接扰动正确排名。
3. **目标书册不等于唯一证据书册。** 系列小说中的事实会在后续书册重复、回顾或补充。例如开发集中：
   - “What kind of creature is Aragog?” 属于 Book 2，Baseline 的完整答案字符串 “Acromantula” 来自 Book 6 passage。
   - “Which Minister of Magic delivered the bequests from Dumbledore's will?” 属于 Book 7，Baseline 的完整答案字符串 “Rufus Scrimgeour” 来自 Book 6 passage。
4. **目标书册占比不是证据质量。** 各路由方案都明显提高了该占比，却没有同步提高 Contain-Acc；把结果限制得更“同册”不代表答案证据更好。
5. **图检索本身依赖跨册连接。** LinearRAG 会利用跨书册重复实体和语义桥接传播重要性，额外书册权重可能与这些有用连接发生冲突。
6. **Contain-Acc 有局限。** 它只判断答案字符串是否出现，可能奖励另一书册中的重复表述，也不能证明 passage 完整支持答案；但在同一批问题的配对实验中，三种预测路由均未稳定改善 Top-10/20 或 MRR，结论方向是一致的。

## 项目决策

本轮到此结束：不继续做参数网格、不测试 rank fusion，也不运行昂贵的 LLM 回答评测。后续若需要回访这个方向，应先重新定义“问题所属书册”和“证据所在书册”的标注，或训练更可靠的书册预测器，而不是继续放大路由权重。

原始机器可读结果位于本地：

```text
results/book_routing/dev_m3_t005_s050_2026-09-13/
results/book_routing/dev_m3_t005_s025_2026-09-13/
results/book_routing/dev_boostonly_m3_t005_s050_2026-09-13/
```
