# Narrative-aware LinearRAG：四个独立任务分工

> 目标：将当前方案拆分为 **3 个创新任务 + 1 个独立对照实验任务**。  
> 本版本**不包含最终模块合并、Full Model 集成、统一收尾等工作**，四个任务均按可独立推进的方式描述。

---

## 任务 1：实体别名扩展  
### Alias-aware Entity Activation

### 任务目标

解决长篇小说中**同一实体存在多个名字、称呼或身份名**时，检索无法完整命中的问题。

例如：

- Voldemort
- Tom Riddle
- You-Know-Who
- Dark Lord

这些名称实际上指向同一个人物。如果查询只出现其中一个名字，希望系统能够同步激活该实体的其他别名及相关证据。

### 核心思路

建立：

```text
alias → canonical entity
```

例如：

```text
Tom Riddle
You-Know-Who
Dark Lord
        ↓
    Voldemort
```

查询阶段遇到任意别名后，都统一映射到同一个实体 ID，并进行别名扩展检索。

### 主要工作

1. 阅读并定位 Original LinearRAG 中：
   - NER
   - entity extraction
   - entity indexing
   - query entity activation

2. 为实体设计别名结构，例如：

```text
Entity {
    canonical_name
    aliases[]
    entity_id
}
```

3. 构造小说中的实体别名信息：
   - 可利用 NER / LLM 自动抽取
   - 再进行少量人工校验

4. 查询阶段实现：
   - alias normalization
   - alias expansion
   - canonical entity activation

5. 尽量保持原 LinearRAG 主流程不变，以独立模块形式实现。

### 需要验证的问题

核心问题：

> 加入实体别名后，是否能够找回原本因为称呼不同而遗漏的实体和证据？

重点测试：

- 别名问题
- 身份名问题
- 不同时间段人物称呼变化的问题

### 主要指标

- Entity Recall
- Evidence Recall@k
- Answer Accuracy

### 最终交付物

- Alias-aware Entity Activation 模块
- 实体别名数据
- 典型测试案例
- `Original LinearRAG vs +Alias` 实验结果

---

## 任务 2：书册感知软路由  
### Book-aware Soft Routing

### 任务目标

解决多册长篇小说混合检索时，**其他书册中的相关人物或事件对当前问题产生干扰**的问题。

例如，一个问题主要发生在第三册，但某个角色在七册中都出现。

Original LinearRAG 可能从所有书中返回相关文本，从而引入：

- 错误时间线
- 错误事件
- 跨册噪声

### 核心思路

先根据问题判断：

> 这个问题更可能与哪一册相关？

得到类似：

```text
Book 1 : 0.05
Book 2 : 0.10
Book 3 : 0.70
Book 4 : 0.10
Book 5 : 0.05
```

然后把这个结果作为**软先验**加入检索排序。

注意：

**不是直接只搜某一本书。**

应采用 Soft Routing，而不是 Hard Filtering。

### 可能形式

例如原始 passage 得分为：

\[
S_{rag}(p)
\]

书册相关度为：

\[
S_{book}(p)
\]

最终可以设计：

\[
S(p)=S_{rag}(p)+\lambda S_{book}(p)
\]

具体融合方法可以通过实验决定。

### 主要工作

1. 给所有 chunk / passage 添加：
   - `book_id`
   - book metadata

2. 设计 Query → Book relevance 的方法，例如：
   - embedding similarity
   - LLM classification
   - book-level representation

3. 得到书册概率或书册相关度。

4. 将 book score 融入 Original LinearRAG 的 retrieval score。

5. 保留 global fallback：
   - 防止预测书册错误后完全搜不到正确答案
   - 支持真正的跨册问题

### 需要验证的问题

核心问题：

> Book-aware routing 是否真的能够减少跨册噪声，同时不显著损失正确证据召回？

重点测试：

- 同一人物跨多册出现
- 相似事件在多册重复出现
- 明确属于某一册的问题
- 真正需要跨册信息的问题

### 主要指标

- Evidence Recall@k
- Answer Accuracy
- 跨册错误证据比例
- Latency

### 最终交付物

- Book-aware Soft Routing 模块
- Book metadata / book representation
- Routing 实现
- `Original LinearRAG vs +Book` 实验结果

---

## 任务 3：章节级粗到细检索  
### Chapter-level Coarse-to-Fine Retrieval

### 任务目标

解决 Original LinearRAG 直接在全书 passage 中搜索时，返回结果可能：

- 过于零散
- 分布在很多章节
- 缺乏完整叙事上下文

的问题。

### 核心思路

将原来的：

```text
Question
   ↓
All Passages
   ↓
Top-k Evidence
```

修改为：

```text
Question
   ↓
Relevant Chapters
   ↓
Passages inside selected chapters
   ↓
Top-k Evidence
```

即：

> Chapter-level Coarse Retrieval  
> → Passage-level Fine Retrieval

### Chapter 表示方式

可以尝试：

#### 方案 A：章节摘要表示

```text
Chapter Title
+
Chapter Summary
        ↓
    Embedding
```

#### 方案 B：章节内部信息聚合

对该章节的：

- passages
- entities
- embeddings

进行 aggregation，形成 chapter representation。

两种方案可以先选择简单方案实现，再视时间决定是否比较。

### 主要工作

1. 对小说进行 chapter 划分。

2. 建立：
   - chapter_id
   - chapter metadata
   - chapter representation

3. 建立章节级索引。

4. Query 首先检索 Top-N Chapters。

5. 然后只在候选章节中执行 passage retrieval。

6. 设计 fallback：
   - 防止第一阶段章节召回错误
   - 可以保留少量 global retrieval 候选

### 需要验证的问题

核心问题：

> 章节级 coarse-to-fine 是否能够让检索结果更加集中在正确叙事区间，同时保持或提高证据召回？

重点观察：

- 原 LinearRAG 是否把 evidence 分散到很多章节
- Chapter-aware 方法是否更容易找到连续上下文
- 对跨章节问题是否会产生负面影响

### 主要指标

- Evidence Recall@k
- Answer Accuracy
- 检索证据所在章节数量
- Retrieval Latency

### 最终交付物

- Chapter-level index
- Coarse-to-Fine Retrieval 模块
- 章节表示方法
- 典型 case study
- `Original LinearRAG vs +Chapter` 实验结果

---

## 任务 4：Baseline 与统一对照实验  
### Baseline & Evaluation

> 本任务只负责**建立可靠的对照实验和评测体系**，  
> **不负责三个创新模块最终合并，也不负责 Full Model 集成。**

### 任务目标

建立统一实验环境，用于判断后续改进是否真正有效。

需要至少跑通三个 Baseline：

```text
Closed-book
Dense RAG
Original LinearRAG
```

其中 Original LinearRAG 是三个创新任务最核心的比较对象。

### Baseline 1：Closed-book

不给模型任何检索内容，仅使用：

```text
Question
→ LLM
→ Answer
```

作用：

> 判断模型自身知识能够回答多少问题。

---

### Baseline 2：Dense RAG

使用普通向量检索：

```text
Question
→ Embedding
→ Vector Retrieval
→ Top-k Passages
→ LLM
```

作用：

> 判断 LinearRAG 相比最基础 RAG 是否确实具有优势。

---

### Baseline 3：Original LinearRAG

严格按照原论文 / 原代码运行。

作用：

> 作为三个创新模块的直接基线。

三个创新任务的结果应分别能够与它进行比较：

```text
Original LinearRAG
vs
+ Alias
```

```text
Original LinearRAG
vs
+ Book
```

```text
Original LinearRAG
vs
+ Chapter
```

---

## 实验控制变量

为了保证比较公平，需要统一：

- Same Generator / LLM
- Same Question Set
- Same Prompt
- Same Top-k
- Same chunk 设置
- Same generation parameters

除被研究变量外，其他设置尽量保持一致。

---

## 评测指标

建议至少实现以下指标：

### 1. Answer Accuracy

最终答案是否正确。

---

### 2. Contain-Acc

模型输出是否包含标准答案中的关键内容。

适用于开放式生成答案。

---

### 3. Evidence Recall@k

设标准证据集合为：

\[
E_{gold}
\]

Top-k 检索结果为：

\[
E_k
\]

则评估：

> 正确证据是否被检索进入 Top-k。

这是三个 retrieval 改进非常重要的指标。

---

### 4. Latency

记录：

- Retrieval Latency
- Total QA Latency

防止出现：

> Accuracy 提升很小，但检索成本大幅增加。

---

## 重要实验规范：禁止泄露 Gold Book Label

数据集中可以保存：

```text
question
answer
gold evidence
gold book
gold chapter
```

但是：

**gold book / gold chapter 只能用于评测，不能作为正式检索输入。**

错误做法：

```text
Question
+ “This question belongs to Book 3”
→ Retriever
```

正确做法：

```text
Question
→ System predicts relevant book/chapter
→ Retrieval
```

否则 Book-aware / Chapter-aware 方法就失去了实验意义。

---

## 主要工作

1. 整理统一 Question Set。
2. 准备标准答案。
3. 准备 gold evidence。
4. 准备 gold book / chapter 标签，仅供评测使用。
5. 跑通 Closed-book。
6. 跑通 Dense RAG。
7. 跑通 Original LinearRAG。
8. 实现统一 evaluation script。
9. 输出统一格式的结果表。

### 最终交付物

- Question Set
- Gold Answer / Evidence 数据
- Closed-book baseline
- Dense RAG baseline
- Original LinearRAG baseline
- Evaluation scripts
- Baseline 实验结果表

---

# 四个任务简表

| 任务 | 名称 | 核心问题 | 主要比较 |
|---|---|---|---|
| 1 | Alias-aware Entity Activation | 同一人物多名称导致实体召回不足 | Original vs +Alias |
| 2 | Book-aware Soft Routing | 多册混合导致跨册噪声 | Original vs +Book |
| 3 | Chapter-level Coarse-to-Fine Retrieval | Passage 过于零散、缺乏章节上下文 | Original vs +Chapter |
| 4 | Baseline & Evaluation | 建立公平可靠的对照与评测体系 | Closed-book / Dense RAG / Original |

---

## 当前阶段不包含的工作

本轮分工**暂不包含**：

- 三个创新模块最终合并
- Full Model 集成
- `+Alias +Book +Chapter` 联合实验
- Full Model 最终消融
- 最终代码整合与冲突处理

这些工作等四个独立任务基本完成后，再单独安排。
