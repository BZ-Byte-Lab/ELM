# ELM 项目工作流详解

本文档详细介绍 ELM (Embedding Language Model) 项目的完整工作流程。

## 目录

1. [项目概述](#1-项目概述)
2. [第一阶段：数据准备 (Data_Preparation)](#2-第一阶段数据准备-data_preparation)
3. [第二阶段：数据合成 (Data_Synthesis)](#3-第二阶段数据合成-data_synthesis)
4. [第三阶段：模型训练 (Training)](#4-第三阶段模型训练-training)
5. [数据流向图](#5-数据流向图)
6. [输出格式说明](#6-输出格式说明)
7. [运行指南](#7-运行指南)
8. [项目级工具脚本](#8-项目级工具脚本)
9. [总结](#9-总结)

---

## 1. 项目概述

### 1.1 项目目标

ELM 项目是一个三阶段数据处理和模型训练流水线，用于：

1. **数据准备阶段**：从 WikiText-2 数据集提取高质量文本段落，并使用 Qwen3-Embedding-4B 模型生成 2560 维嵌入向量
2. **数据合成阶段**：利用 LLM (Qwen3-30B-A3B-Instruct) 生成多样化的合成训练数据，覆盖 16 种任务类型
3. **模型训练阶段**：训练 ELM MLP Adapter，学习将嵌入向量映射到 Qwen3-4B-Instruct 的 token 嵌入空间

### 1.2 项目结构

```
elm/
├── Data_Preparation/           # 第一阶段：数据准备
│   ├── data_pipeline/          # 核心处理模块
│   │   ├── config.py           # 配置管理
│   │   ├── download.py         # 数据下载
│   │   ├── preprocess.py       # 文本预处理
│   │   ├── embeddings.py       # 嵌入生成
│   │   ├── dataset.py          # 数据集类
│   │   └── utils.py            # 工具函数
│   └── scripts/                # 执行脚本
│       ├── run_pipeline.py     # 完整流水线
│       └── run_embedding_only.py
│
├── Data_Synthesis/             # 第二阶段：数据合成
│   ├── synthesis_pipeline/     # 核心合成模块
│   │   ├── config.py           # 合成配置
│   │   ├── task_registry.py    # 任务注册表
│   │   ├── knn_index.py        # k-NN 索引
│   │   ├── api_client.py       # API 客户端
│   │   ├── quality_filter.py   # 质量过滤
│   │   ├── checkpoint.py       # 检查点系统
│   │   ├── generator.py        # 主生成器
│   │   ├── output_writer.py    # JSONL 输出写入
│   │   ├── validator.py        # 验证检查清单
│   │   ├── profiler.py         # 性能分析器
│   │   └── utils.py            # 工具函数
│   └── scripts/
│       └── run_synthesis.py    # 合成入口
│
├── Training/                   # 第三阶段：模型训练
│   ├── environment.yml         # 独立 Conda 环境
│   ├── pyproject.toml          # 包配置
│   ├── training_pipeline/      # 核心训练模块
│   │   ├── config.py           # 训练配置
│   │   ├── adapter.py          # MLP Adapter
│   │   ├── model.py            # ELM 模型
│   │   ├── dataset.py          # 训练数据集
│   │   ├── trainer.py          # 训练器
│   │   ├── checkpoint.py       # 检查点管理
│   │   └── utils.py            # 工具函数
│   └── scripts/
│       ├── train.py            # 训练入口
│       └── evaluate.py         # 评估脚本
│
├── scripts/                    # 项目级脚本
│   ├── sort_by_task_type.py    # 按任务类型排序 JSONL
│   └── update_checkpoint.py    # 更新检查点文件
│
├── data/                       # 输出数据目录
│   ├── wikitext2_processed/    # 预处理文本 (Parquet)
│   ├── embeddings/             # 嵌入向量 (SafeTensors)
│   ├── synthesis/              # 合成数据 (JSONL)
│   └── checkpoints/            # Adapter 检查点
│
└── logs/                       # 日志文件
```

---

## 2. 第一阶段：数据准备 (Data_Preparation)

数据准备阶段负责将原始 WikiText-2 数据转换为带有嵌入向量的高质量文本段落。

### 2.1 配置模块 (config.py)

配置模块定义了整个数据准备流程的参数：

#### 路径配置
```python
data_dir = "/home/benz/coding_project/elm/data"
processed_dir = "data/wikitext2_processed/"   # Parquet 输出
embeddings_dir = "data/embeddings/"           # SafeTensors 输出
```

#### 数据集参数
| 参数 | 值 | 说明 |
|------|-----|------|
| `hf_dataset` | Salesforce/wikitext | HuggingFace 数据集名称 |
| `hf_config` | wikitext-2-v1 | 数据集版本 |
| `min_tokens` | 100 | 最小 token 数 |
| `max_tokens` | 2000 | 最大 token 数 |
| `train_ratio` | 0.8 | 训练集比例 |
| `val_ratio` | 0.1 | 验证集比例 |
| `test_ratio` | 0.1 | 测试集比例 |
| `random_seed` | 42 | 随机种子（可复现性） |

#### 嵌入模型参数
| 参数 | 值 | 说明 |
|------|-----|------|
| `model_name` | Qwen/Qwen3-Embedding-4B | 嵌入模型 |
| `embedding_dim` | 2560 | 嵌入维度 |
| `max_length` | 8192 | 最大序列长度 |
| `batch_size` | 8 | 批处理大小（16GB VRAM） |
| `use_flash_attention` | True | Flash Attention 2 加速 |
| `use_fp16` | True | 半精度推理 |

### 2.2 数据下载 (download.py)

数据下载模块从 HuggingFace 加载 WikiText-2 数据集。

#### 函数：`load_wikitext2(config)`

```python
from datasets import load_dataset

def load_wikitext2(config):
    """
    加载 WikiText-2 数据集

    返回:
        DatasetDict: 包含 train, validation, test 三个切分
    """
    dataset = load_dataset(
        config.hf_dataset,      # "Salesforce/wikitext"
        config.hf_config        # "wikitext-2-v1"
    )
    return dataset
```

#### 数据集统计
- **训练集**：约 36,718 行
- **验证集**：约 3,760 行
- **测试集**：约 4,358 行

### 2.3 文本预处理 (preprocess.py)

预处理模块是数据准备的核心，包含多个处理步骤：

#### 步骤 1：段落提取 `extract_paragraphs(dataset)`

将连续的非空行合并为段落：

```python
def extract_paragraphs(dataset):
    """
    从 WikiText-2 提取段落

    处理逻辑：
    1. 遍历所有行
    2. 遇到空行时，将累积的行合并为一个段落
    3. 非空行添加到当前段落缓冲区
    """
    paragraphs = []
    current_paragraph = []

    for line in dataset['text']:
        if line.strip():
            current_paragraph.append(line)
        elif current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []

    return paragraphs
```

#### 步骤 2：文本清洗 `clean_text(text)`

清除 Wikipedia 格式残留：

| 清洗规则 | 示例 | 说明 |
|---------|------|------|
| 移除章节标题 | `= = Section = =` → 移除 | Wikipedia 标题格式 |
| 移除特殊标记 | `@-@`, `@.@`, `@,@` → 移除 | 分词残留 |
| 移除未知词 | `<unk>` → 移除 | 词汇表外词 |
| 合并空白 | 多个空格 → 单个空格 | 标准化 |
| 去除首尾空白 | `strip()` | 清理 |

```python
def clean_text(text):
    # 移除 Wikipedia 章节标题
    text = re.sub(r'\s*=\s*=.*?=\s*=\s*', '', text)
    # 移除特殊标记
    text = text.replace('@-@', '').replace('@.@', '').replace('@,@', '')
    # 移除 <unk>
    text = text.replace('<unk>', '')
    # 合并空白
    text = ' '.join(text.split())
    return text.strip()
```

#### 步骤 3：质量检查 `is_low_quality(text)`

过滤低质量文本：

```python
def is_low_quality(text):
    """
    判断文本是否为低质量

    过滤条件：
    - 特殊字符占比 > 50%
    - 数字占比 > 50%
    - 文本长度 < 20 字符
    """
    if len(text) < 20:
        return True

    special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
    digit_ratio = sum(1 for c in text if c.isdigit()) / len(text)

    return special_ratio > 0.5 or digit_ratio > 0.5
```

#### 步骤 4：Token 过滤 `filter_paragraphs(paragraphs, tokenizer, min_tokens, max_tokens)`

基于 token 数量筛选段落：

```python
def filter_paragraphs(paragraphs, tokenizer, min_tokens=100, max_tokens=2000):
    """
    按 token 数量过滤段落

    返回:
        List[Dict]: 包含 text, token_count, char_count 的字典列表
    """
    filtered = []

    for text in paragraphs:
        # 使用 Qwen tokenizer 计算 token 数
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_count = len(tokens)

        if min_tokens <= token_count <= max_tokens:
            filtered.append({
                'text': text,
                'token_count': token_count,
                'char_count': len(text)
            })

    return filtered
```

**典型保留率**：约 5-15% 的原始段落通过过滤

#### 步骤 5：数据集切分 `split_dataset(data, train_ratio, val_ratio, test_ratio, seed)`

```python
def split_dataset(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    切分数据集为 train/val/test

    处理：
    1. 使用固定种子打乱数据（可复现性）
    2. 按比例切分
    3. 添加唯一 text_id
    """
    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # 添加 text_id
    for i, item in enumerate(data[:train_end]):
        item['text_id'] = f'train_{i}'
    # ... val 和 test 类似

    return train_data, val_data, test_data
```

#### 步骤 6：保存为 Parquet `save_processed_data(train, val, test, config)`

```python
def save_processed_data(train_data, val_data, test_data, config):
    """
    使用 Polars 保存为 Parquet 格式

    输出文件：
    - train.parquet
    - val.parquet
    - test.parquet
    """
    import polars as pl

    for split_name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        df = pl.DataFrame(data)
        df.write_parquet(config.processed_dir / f'{split_name}.parquet')
```

### 2.4 嵌入向量生成 (embeddings.py)

嵌入模块使用 Qwen3-Embedding-4B 模型生成文本嵌入。

#### 步骤 1：加载模型 `load_embedding_model(config)`

```python
def load_embedding_model(config):
    """
    加载 Qwen3-Embedding-4B 模型

    优化配置：
    - Flash Attention 2（如果可用且有 CUDA）
    - FP16 半精度（节省显存）
    - Left-padding（适配 last-token pooling）
    """
    model_kwargs = {}

    if config.use_flash_attention and torch.cuda.is_available():
        model_kwargs['attn_implementation'] = 'flash_attention_2'

    if config.use_fp16:
        model_kwargs['torch_dtype'] = torch.float16

    model = AutoModel.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        **model_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        padding_side='left'  # 重要：适配 last-token pooling
    )

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    return model, tokenizer
```

#### 步骤 2：Last-Token Pooling `last_token_pool(last_hidden_states, attention_mask)`

Qwen3-Embedding 使用 **last-token pooling** 提取句子嵌入：

```python
def last_token_pool(last_hidden_states, attention_mask):
    """
    从最后一个非 padding token 提取嵌入

    原理：
    - 对于 left-padding：最后一个位置就是最后一个 token
    - 对于 right-padding：需要找到最后一个非 0 的 attention mask 位置

    参数：
        last_hidden_states: (batch_size, seq_len, hidden_dim)
        attention_mask: (batch_size, seq_len)

    返回：
        embeddings: (batch_size, hidden_dim)
    """
    # 找到每个序列最后一个有效 token 的位置
    sequence_lengths = attention_mask.sum(dim=1) - 1

    batch_size = last_hidden_states.shape[0]

    # 提取对应位置的隐藏状态
    embeddings = last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths
    ]

    return embeddings
```

**为什么使用 Last-Token Pooling？**

| 方法 | 说明 | 适用场景 |
|------|------|---------|
| Mean Pooling | 所有 token 隐藏状态的平均 | BERT 类模型 |
| CLS Pooling | 使用 [CLS] token | BERT 类模型 |
| **Last-Token Pooling** | 使用最后一个 token | **Qwen3-Embedding (推荐)** |

Qwen3-Embedding 模型经过专门训练，使得最后一个 token 的隐藏状态包含整个序列的语义信息。

#### 步骤 3：批量生成嵌入 `generate_embeddings_batch(model, tokenizer, texts, config)`

```python
def generate_embeddings_batch(model, tokenizer, texts, config):
    """
    批量生成嵌入向量

    处理流程：
    1. 按 batch_size 分批
    2. Tokenize 并填充
    3. 模型推理
    4. Last-token pooling
    5. L2 归一化

    返回：
        np.ndarray: (n_texts, 2560)
    """
    all_embeddings = []

    for i in tqdm(range(0, len(texts), config.batch_size)):
        batch_texts = texts[i:i + config.batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors='pt'
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # 推理
        with torch.no_grad():
            outputs = model(**inputs)

        # Last-token pooling
        embeddings = last_token_pool(
            outputs.last_hidden_state,
            inputs['attention_mask']
        )

        # L2 归一化（重要：用于余弦相似度）
        embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)
```

**为什么需要 L2 归一化？**

```
归一化后：||e|| = 1

余弦相似度 = e1 · e2 / (||e1|| × ||e2||) = e1 · e2

归一化后，内积 = 余弦相似度，简化 k-NN 计算
```

#### 步骤 4：保存嵌入 `save_embeddings(embeddings, output_path, metadata)`

```python
def save_embeddings(embeddings, output_path, metadata):
    """
    使用 SafeTensors 格式保存嵌入

    SafeTensors 优势：
    - 内存映射支持
    - 快速加载
    - 安全（无代码执行风险）
    """
    from safetensors.numpy import save_file

    tensors = {'embeddings': embeddings}
    save_file(tensors, output_path, metadata=metadata)
```

### 2.5 数据集类 (dataset.py)

`ELMDataset` 类提供 PyTorch Dataset 接口：

```python
class ELMDataset:
    """
    ELM 数据集类

    功能：
    - 加载预处理文本（Parquet）
    - 可选加载嵌入向量（SafeTensors）
    - 支持索引访问和批量加载
    - 嵌入插值功能
    """

    def __init__(self, data_dir, split='train', load_embeddings=True):
        self.data_dir = Path(data_dir)
        self.split = split

        # 加载文本数据
        self.df = pl.read_parquet(
            self.data_dir / 'wikitext2_processed' / f'{split}.parquet'
        )

        # 可选加载嵌入
        self.embeddings = None
        if load_embeddings:
            from safetensors.numpy import load_file
            embeddings_path = self.data_dir / 'embeddings' / f'{split}_embeddings.safetensors'
            self.embeddings = load_file(embeddings_path)['embeddings']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        返回单个样本

        返回：
            Dict: {
                'text': str,
                'embedding': np.ndarray (2560,) 或 None,
                'metadata': {
                    'text_id': str,
                    'token_count': int,
                    'char_count': int
                }
            }
        """
        row = self.df.row(idx, named=True)

        return {
            'text': row['text'],
            'embedding': self.embeddings[idx] if self.embeddings is not None else None,
            'metadata': {
                'text_id': row['text_id'],
                'token_count': row['token_count'],
                'char_count': row['char_count']
            }
        }

    def interpolate_embeddings(self, idx1, idx2, alpha):
        """
        嵌入插值

        公式：e_interp = normalize(alpha * e1 + (1-alpha) * e2)

        用途：生成两个文本之间的"概念中点"
        """
        e1 = self.embeddings[idx1]
        e2 = self.embeddings[idx2]

        interpolated = alpha * e1 + (1 - alpha) * e2

        # 重新 L2 归一化
        interpolated = interpolated / np.linalg.norm(interpolated)

        return interpolated

    def get_dataloader(self, batch_size=32, shuffle=True):
        """创建 PyTorch DataLoader"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )
```

---

## 3. 第二阶段：数据合成 (Data_Synthesis)

数据合成阶段使用 LLM 生成多样化的训练目标。

### 3.1 合成配置 (config.py)

#### 任务类别枚举

```python
class TaskCategory(Enum):
    FACTUAL = "factual"         # 事实类：T=0.3, top_p=0.85
    DESCRIPTIVE = "descriptive" # 描述类：T=0.5, top_p=0.9
    CREATIVE = "creative"       # 创意类：T=0.7, top_p=0.92
    PAIR_BASED = "pair_based"   # 配对类：需要 k-NN
```

#### 合成配置类

```python
@dataclass
class SynthesisConfig:
    # API 设置
    api_base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = "qwen/qwen3-30b-a3b-instruct-2507"

    # 速率限制
    requests_per_minute: int = 60           # 每分钟请求数
    requests_per_second: float = 10         # 每秒请求数
    max_concurrent_requests: int = 10       # 最大并发请求数（异步模式）
    max_retries: int = 3                    # 最大重试次数
    retry_delay: float = 1.0                # 重试延迟（秒）

    # k-NN 设置
    knn_k: int = 10                         # k-NN 邻居数
    knn_metric: str = "cosine"              # 相似度度量
    use_gpu_index: bool = False             # GPU 加速 FAISS

    # 生成设置
    checkpoint_interval: int = 20           # 检查点间隔
    variations_per_task: int = 2            # 每任务变体数
    min_samples_per_embedding: int = 15     # 每个嵌入最小样本数
    batch_size: int = 50                    # 批处理大小（异步操作）

    # 异步处理模式
    use_async: bool = True                  # 启用异步处理
    max_concurrent_batches: int = 3         # 最大并发批次数

    # 性能分析
    enable_profiling: bool = True           # 启用性能分析追踪

    # 质量控制
    max_rejection_rate: float = 0.20        # 最大拒绝率
    repetition_threshold: float = 0.5       # 重复阈值

    # 嵌入维度（与 Data_Preparation 一致）
    embedding_dim: int = 2560

    # 随机种子（可复现性）
    random_seed: int = 42
```

### 3.2 任务注册表 (task_registry.py)

#### 16 种任务类型详解

##### A. 事实类任务 (Factual) - Temperature=0.3, top_p=0.85

| 任务 | 说明 | 最小 Token | Prompt 模板 |
|------|------|-----------|-------------|
| `keywords` | 提取 5-7 个关键概念 | 25 | "Extract 5-7 key concepts from this text..." |
| `category` | 识别学术领域 | 40 | "Identify the academic field or domain..." |
| `questions` | 生成 3 个可回答问题 | 60 | "Generate 3 questions that can be answered..." |

##### B. 描述类任务 (Descriptive) - Temperature=0.5, top_p=0.9

| 任务 | 说明 | 最小 Token | Prompt 模板 |
|------|------|-----------|-------------|
| `summary` | 2-3 句摘要 | 50 | "Write a concise 2-3 sentence summary..." |
| `describe` | 详细描述 | 120 | "Provide a detailed description..." |
| `explain_beginner` | 初学者解释 | 100 | "Explain this content for a beginner..." |
| `explain_expert` | 专家解释 | 100 | "Explain this content for an expert..." |
| `related_topics` | 5 个相关主题 | 80 | "List 5 related topics with connections..." |

##### C. 创意类任务 (Creative) - Temperature=0.7, top_p=0.92

| 任务 | 说明 | 最小 Token | Prompt 模板 |
|------|------|-----------|-------------|
| `characteristics_pos` | 5 个优点/亮点 | 80 | "List 5 strengths or interesting aspects..." |
| `characteristics_neg` | 5 个局限/批评 | 80 | "List 5 limitations or criticisms..." |
| `style_academic` | 学术风格重写 | 100 | "Rewrite in formal academic style..." |
| `style_casual` | 口语风格重写 | 100 | "Rewrite in casual conversational style..." |
| `counterfactual` | 应用到随机领域 | 100 | "Apply these concepts to {random_domain}..." |

**Counterfactual 任务的 23 个随机领域**：
```python
DOMAINS = [
    "cooking", "sports", "music", "architecture", "fashion",
    "gaming", "gardening", "photography", "travel", "psychology",
    "economics", "politics", "medicine", "engineering", "art",
    "literature", "philosophy", "astronomy", "biology", "chemistry",
    "physics", "mathematics", "computer science"
]
```

##### D. 配对类任务 (Pair-Based) - 需要 k-NN

| 任务 | 说明 | k值 | Alpha | 最小 Token | Temperature |
|------|------|-----|-------|-----------|-------------|
| `compare` | 比较两个相似文本 | 3 | - | 150 | 0.5 |
| `hypothetical` | 概念中点描述 | 2 | 0.3-0.7 | 120 | 0.7 |

#### TaskConfig 数据类

```python
@dataclass
class TaskConfig:
    name: str                    # 任务名称
    category: TaskCategory       # 任务类别
    prompt_template: str         # Prompt 模板
    min_tokens: int              # 最小输出 token 数
    temperature: float           # 生成温度
    top_p: float                 # Top-p 采样
    requires_pair: bool = False  # 是否需要配对
    k_neighbors: int = 0         # k-NN 邻居数（配对任务）
    alpha_range: tuple = None    # Alpha 范围（hypothetical 任务）
```

#### TaskRegistry 类

```python
class TaskRegistry:
    """任务注册表"""

    def __init__(self):
        self.tasks = {}
        self._register_all_tasks()

    def _register_all_tasks(self):
        # 注册所有 16 种任务
        self._register_factual_tasks()
        self._register_descriptive_tasks()
        self._register_creative_tasks()
        self._register_pair_tasks()

    def get_single_text_tasks(self):
        """获取所有单文本任务（14 种）"""
        return [t for t in self.tasks.values() if not t.requires_pair]

    def get_pair_tasks(self):
        """获取所有配对任务（2 种）"""
        return [t for t in self.tasks.values() if t.requires_pair]

    def get_tasks_by_category(self, category: TaskCategory):
        """按类别获取任务"""
        return [t for t in self.tasks.values() if t.category == category]
```

### 3.3 k-NN 索引 (knn_index.py)

k-NN 索引用于查找语义相似的文本，支持配对任务。

#### FAISS 索引构建

```python
class KNNIndex:
    """
    基于 FAISS 的 k-NN 索引

    使用 IndexFlatIP（内积）而非 IndexFlatL2：
    - 嵌入已 L2 归一化
    - 内积 = 余弦相似度
    - 更高效
    """

    def __init__(self, embeddings: np.ndarray, use_gpu: bool = False):
        self.embeddings = embeddings.astype(np.float32)
        self.dimension = embeddings.shape[1]

        # 创建 FAISS 索引
        self.index = faiss.IndexFlatIP(self.dimension)

        # 可选 GPU 加速
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        # 添加嵌入
        self.index.add(self.embeddings)

    def search(self, query_idx: int, k: int):
        """
        搜索 k 个最近邻（排除自身）

        参数：
            query_idx: 查询嵌入的索引
            k: 返回的邻居数

        返回：
            neighbor_indices: 邻居索引数组
            similarities: 相似度分数数组
        """
        query = self.embeddings[query_idx:query_idx+1]

        # 搜索 k+1 个（包含自身）
        similarities, indices = self.index.search(query, k + 1)

        # 过滤掉自身
        mask = indices[0] != query_idx
        neighbor_indices = indices[0][mask][:k]
        neighbor_similarities = similarities[0][mask][:k]

        return neighbor_indices, neighbor_similarities

    def interpolate_embeddings(self, idx1: int, idx2: int, alpha: float):
        """
        嵌入插值

        公式：e_interp = normalize(alpha * e1 + (1-alpha) * e2)
        """
        e1 = self.embeddings[idx1]
        e2 = self.embeddings[idx2]

        interpolated = alpha * e1 + (1 - alpha) * e2
        interpolated = interpolated / np.linalg.norm(interpolated)

        return interpolated
```

#### k-NN 在配对任务中的应用

```
Compare 任务流程：
1. 对于嵌入 i，查找 k=3 个最近邻
2. 对每个邻居 j：
   - 获取文本 text_i 和 text_j
   - 生成 prompt: "Compare these two texts: [text_i] vs [text_j]"
   - 调用 LLM 生成比较结果

Hypothetical 任务流程：
1. 对于嵌入 i，查找 k=2 个最近邻
2. 对每个邻居 j：
   - 生成随机 alpha ∈ [0.3, 0.7]
   - 计算插值嵌入：e_interp = alpha * e_i + (1-alpha) * e_j
   - 生成 prompt: "Describe the conceptual midpoint..."
   - 调用 LLM 生成描述
```

### 3.4 API 客户端 (api_client.py)

API 客户端负责与 OpenRouter API 通信。

#### 双层速率限制系统

```python
class AsyncRateLimiter:
    """
    异步速率限制器（令牌桶算法）

    特点：
    - 事件驱动的令牌补充
    - 信号量控制最大并发
    - 无轮询延迟
    """

    def __init__(self, requests_per_second: float, max_concurrent: int):
        self.rate = requests_per_second
        self.tokens = requests_per_second
        self.max_tokens = requests_per_second
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def acquire(self):
        """获取一个令牌"""
        async with self.semaphore:  # 限制并发数
            async with self.lock:
                # 补充令牌
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(
                    self.max_tokens,
                    self.tokens + elapsed * self.rate
                )
                self.last_update = now

                # 等待令牌可用
                if self.tokens < 1:
                    wait_time = (1 - self.tokens) / self.rate
                    await asyncio.sleep(wait_time)
                    self.tokens = 0
                else:
                    self.tokens -= 1
```

#### OpenRouter 客户端

```python
class OpenRouterClient:
    """
    OpenRouter API 客户端

    功能：
    - 同步/异步双模式
    - 自动重试（指数退避）
    - 延迟追踪
    """

    def __init__(self, config: SynthesisConfig):
        self.config = config
        self.api_key = os.environ.get('OPENROUTER_API_KEY')

        # 使用 OpenAI SDK（兼容 OpenRouter）
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=config.api_base_url
        )

        # 异步客户端
        self.async_client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=config.api_base_url
        )

        # 速率限制器
        self.rate_limiter = AsyncRateLimiter(
            config.requests_per_second,
            config.max_concurrent_requests
        )

        # 延迟追踪（最近 100 次请求）
        self.latencies = deque(maxlen=100)

    async def generate_async(self, prompt: str, task_config: TaskConfig):
        """
        异步生成

        流程：
        1. 获取速率限制令牌
        2. 调用 API（带重试）
        3. 记录延迟
        4. 返回结果
        """
        await self.rate_limiter.acquire()

        for attempt in range(self.config.max_retries):
            try:
                start_time = time.monotonic()

                response = await self.async_client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=task_config.temperature,
                    top_p=task_config.top_p,
                    max_tokens=1024
                )

                latency = time.monotonic() - start_time
                self.latencies.append(latency)

                text = response.choices[0].message.content
                token_count = len(text.split())

                return GenerationResult(
                    success=True,
                    text=text,
                    token_count=token_count,
                    latency=latency
                )

            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    # 指数退避
                    await asyncio.sleep(2 ** attempt)
                else:
                    return GenerationResult(
                        success=False,
                        error=str(e)
                    )

    async def generate_batch_async(self, prompts: List[str], task_config: TaskConfig):
        """
        异步批量生成

        使用 asyncio.gather 并行处理
        """
        tasks = [
            self.generate_async(prompt, task_config)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)
```

### 3.5 质量过滤 (quality_filter.py)

质量过滤模块确保生成内容的质量。

#### 5 层过滤检查

```python
class QualityFilter:
    """
    质量过滤器

    5 层检查：
    1. 最小 token 数
    2. 指令泄露检测
    3. 重复内容检测
    4. 无意义内容检测
    5. 与原文相似度检测
    """

    def __init__(self, config: SynthesisConfig):
        self.config = config
        self.rejection_counts = defaultdict(int)
        self.total_counts = defaultdict(int)

        # 指令泄露关键词
        self.instruction_patterns = [
            r'\btext:\b',
            r'\bhere is\b',
            r'\bplease\b',
            r'\bprovide\b',
            r'\bwrite\b',
            r'\bgenerate\b'
        ]

    def filter(self, text: str, original_text: str, task_config: TaskConfig) -> tuple:
        """
        执行过滤

        返回：
            (passed: bool, reason: str or None)
        """
        self.total_counts[task_config.name] += 1

        # 检查 1：最小 token 数
        if len(text.split()) < task_config.min_tokens:
            self.rejection_counts[task_config.name] += 1
            return False, "too_short"

        # 检查 2：指令泄露
        text_lower = text.lower()
        for pattern in self.instruction_patterns:
            if re.search(pattern, text_lower):
                self.rejection_counts[task_config.name] += 1
                return False, "instruction_leakage"

        # 检查 3：重复内容（trigram 频率）
        if self._is_repetitive(text):
            self.rejection_counts[task_config.name] += 1
            return False, "repetitive"

        # 检查 4：无意义内容
        if self._is_nonsensical(text):
            self.rejection_counts[task_config.name] += 1
            return False, "nonsensical"

        # 检查 5：与原文过于相似
        if self._too_similar(text, original_text):
            self.rejection_counts[task_config.name] += 1
            return False, "too_similar"

        return True, None

    def _is_repetitive(self, text: str) -> bool:
        """
        检查重复内容

        方法：计算 trigram 最大频率
        阈值：max_freq / total_trigrams > 0.5
        """
        words = text.split()
        if len(words) < 3:
            return False

        trigrams = [
            tuple(words[i:i+3])
            for i in range(len(words) - 2)
        ]

        trigram_counts = Counter(trigrams)
        max_count = max(trigram_counts.values())

        return max_count / len(trigrams) > self.config.repetition_threshold

    def _is_nonsensical(self, text: str) -> bool:
        """
        检查无意义内容

        条件：
        - 空字符串
        - 字母字符 < 50%
        """
        if not text.strip():
            return True

        alpha_count = sum(1 for c in text if c.isalpha())
        return alpha_count / len(text) < 0.5

    def _too_similar(self, text: str, original: str) -> bool:
        """
        检查与原文相似度

        方法：单词重叠率
        阈值：> 80%
        """
        text_words = set(text.lower().split())
        original_words = set(original.lower().split())

        if not text_words:
            return False

        overlap = len(text_words & original_words) / len(text_words)
        return overlap > 0.8

    def get_rejection_rates(self) -> dict:
        """获取各任务的拒绝率"""
        rates = {}
        for task_name in self.total_counts:
            total = self.total_counts[task_name]
            rejected = self.rejection_counts[task_name]
            rates[task_name] = rejected / total if total > 0 else 0
        return rates

    def check_rejection_threshold(self, max_rate: float = 0.20):
        """检查是否超过拒绝率阈值"""
        rates = self.get_rejection_rates()
        warnings = []
        for task_name, rate in rates.items():
            if rate > max_rate:
                warnings.append(f"{task_name}: {rate:.1%}")
        return warnings
```

### 3.6 检查点系统 (checkpoint.py)

检查点系统支持断点续传和去重。

#### 检查点状态

```python
@dataclass
class CheckpointState:
    """检查点状态"""
    split: str                      # 数据切分（train/val/test）
    task_name: str                  # 任务名称
    completed_indices: Set[int]     # 已完成的嵌入索引
    total_generations: int          # 总生成数
    last_update: str                # 最后更新时间（ISO格式）
    rejection_count: int            # 拒绝数
    metadata: Dict[str, Any]        # 额外元数据
```

#### 检查点管理器

```python
class CheckpointManager:
    """
    检查点管理器

    功能：
    - 保存/加载检查点
    - MD5 哈希去重
    - 异步安全操作
    """

    def __init__(self, checkpoint_dir: Path, interval: int = 20):
        self.checkpoint_dir = checkpoint_dir
        self.interval = interval
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 哈希缓存（用于去重）
        self.hash_cache_path = self.checkpoint_dir / 'hash_cache.json'
        self.hash_cache = self._load_hash_cache()

        # 异步锁
        self.lock = asyncio.Lock()

    def _load_hash_cache(self) -> Set[str]:
        """加载哈希缓存"""
        if self.hash_cache_path.exists():
            with open(self.hash_cache_path) as f:
                return set(json.load(f))
        return set()

    def _save_hash_cache(self):
        """保存哈希缓存"""
        with open(self.hash_cache_path, 'w') as f:
            json.dump(list(self.hash_cache), f)

    def is_duplicate(self, text: str) -> bool:
        """
        检查是否为重复内容

        使用 MD5 哈希
        """
        text_hash = hashlib.md5(text.encode()).hexdigest()

        if text_hash in self.hash_cache:
            return True

        self.hash_cache.add(text_hash)
        return False

    async def is_duplicate_async(self, text: str) -> bool:
        """异步安全的重复检查"""
        async with self.lock:
            return self.is_duplicate(text)

    def load_checkpoint(self, split: str, task_name: str) -> Optional[CheckpointState]:
        """加载检查点"""
        checkpoint_path = self.checkpoint_dir / f'{split}_{task_name}_checkpoint.json'

        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path) as f:
            data = json.load(f)

        return CheckpointState(
            split=data['split'],
            task_name=data['task_name'],
            completed_indices=set(data['completed_indices']),
            total_generations=data['total_generations'],
            last_update=data['last_update'],
            rejection_count=data['rejection_count'],
            metadata=data.get('metadata', {})
        )

    def save_checkpoint(self, state: CheckpointState):
        """保存检查点"""
        checkpoint_path = self.checkpoint_dir / f'{state.split}_{state.task_name}_checkpoint.json'

        data = {
            'split': state.split,
            'task_name': state.task_name,
            'completed_indices': list(state.completed_indices),
            'total_generations': state.total_generations,
            'last_update': datetime.now().isoformat(),
            'rejection_count': state.rejection_count,
            'metadata': state.metadata
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)

        # 同时保存哈希缓存
        self._save_hash_cache()

    def should_checkpoint(self, count: int) -> bool:
        """检查是否应该保存检查点"""
        return count > 0 and count % self.interval == 0
```

### 3.7 输出写入器 (output_writer.py)

输出写入器负责将生成的合成数据写入 JSONL 文件。

#### OutputWriter 类

```python
class OutputWriter:
    """
    JSONL 输出写入器

    功能：
    - 原子性写入操作
    - 自动创建输出目录
    - 支持结构化输出格式
    """

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, output: SynthesisOutput):
        """
        写入单个合成输出

        参数：
            output: SynthesisOutput 数据类实例
        """
        with open(self.output_path, 'a', encoding='utf-8') as f:
            json.dump(output.to_dict(), f, ensure_ascii=False)
            f.write('\n')

    def close(self):
        """关闭写入器（可选，用于资源清理）"""
        pass
```

#### SynthesisOutput 数据类

```python
@dataclass
class SynthesisOutput:
    """合成输出数据结构"""
    task_type: str                   # 任务类型名称
    input_prompt_template: str       # Prompt 模板
    embedding_index: int             # 嵌入索引
    target_text: str                 # 生成的目标文本
    metadata: Dict[str, Any]         # 元数据（温度、top_p、token数等）

    def to_dict(self) -> dict:
        """转换为字典用于 JSON 序列化"""
        return {
            'task_type': self.task_type,
            'input_prompt_template': self.input_prompt_template,
            'embedding_index': self.embedding_index,
            'target_text': self.target_text,
            **self.metadata
        }
```

### 3.8 验证器 (validator.py)

验证器提供全面的输出质量检查清单。

#### Validator 类

```python
class Validator:
    """
    合成数据验证器

    检查项：
    1. 覆盖率：每个嵌入至少有 min_samples 个样本
    2. 配对任务：验证 k-NN 关系
    3. Hypothetical 任务：检查 alpha 范围
    4. 去重：检测重复内容
    5. 拒绝率：确保质量控制在阈值内
    """

    def __init__(self, config: SynthesisConfig):
        self.config = config

    def validate_output(self, split: str) -> ValidationReport:
        """
        验证指定切分的输出

        参数：
            split: 数据切分名称（train/val/test）

        返回：
            ValidationReport: 验证报告
        """
        output_path = self.config.get_synthesis_path(split)

        # 加载所有记录
        records = []
        with open(output_path, 'r') as f:
            for line in f:
                records.append(json.loads(line.strip()))

        report = ValidationReport()

        # 检查 1：覆盖率
        embedding_coverage = defaultdict(set)
        for record in records:
            idx = record['embedding_index']
            task = record['task_type']
            embedding_coverage[idx].add(task)

        min_coverage = min(len(tasks) for tasks in embedding_coverage.values())
        report.min_samples = min_coverage
        report.coverage_ok = min_coverage >= self.config.min_samples_per_embedding

        # 检查 2：配对任务验证
        compare_records = [r for r in records if r['task_type'] == 'compare']
        report.compare_has_neighbors = all('neighbor_idx' in r for r in compare_records)

        # 检查 3：Alpha 范围验证
        hypo_records = [r for r in records if r['task_type'] == 'hypothetical']
        alphas = [r.get('alpha', 0) for r in hypo_records]
        report.alpha_in_range = all(0.3 <= a <= 0.7 for a in alphas)

        # 检查 4：去重
        hashes = set()
        duplicates = 0
        for record in records:
            text_hash = hashlib.md5(record['target_text'].encode()).hexdigest()
            if text_hash in hashes:
                duplicates += 1
            hashes.add(text_hash)

        report.duplicates_found = duplicates
        report.dedup_ok = duplicates == 0

        return report
```

### 3.9 性能分析器 (profiler.py)

性能分析器追踪和记录流水线性能指标。

#### Profiler 类

```python
class Profiler:
    """
    性能分析器

    追踪指标：
    - API 延迟统计
    - 吞吐量（样本/秒）
    - 拒绝率趋势
    - 检查点开销
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.metrics = defaultdict(list)
        self.start_time = time.time()

    def record(self, metric_name: str, value: float):
        """记录单个指标"""
        if not self.enabled:
            return

        self.metrics[metric_name].append({
            'timestamp': time.time(),
            'value': value
        })

    def get_stats(self, metric_name: str) -> dict:
        """
        获取指标统计

        返回：
            {
                'mean': float,
                'median': float,
                'p95': float,
                'p99': float,
                'min': float,
                'max': float,
                'count': int
            }
        """
        if metric_name not in self.metrics:
            return {}

        values = [m['value'] for m in self.metrics[metric_name]]
        values_sorted = sorted(values)

        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }

    def log_summary(self):
        """输出性能摘要到日志"""
        if not self.enabled:
            return

        elapsed = time.time() - self.start_time
        logger.info(f"\n性能分析摘要 (运行时间: {elapsed:.1f}s)")

        for metric_name in self.metrics:
            stats = self.get_stats(metric_name)
            logger.info(f"\n{metric_name}:")
            logger.info(f"  平均: {stats['mean']:.3f}")
            logger.info(f"  中位数: {stats['median']:.3f}")
            logger.info(f"  P95: {stats['p95']:.3f}")
            logger.info(f"  P99: {stats['p99']:.3f}")
            logger.info(f"  样本数: {stats['count']}")
```

### 3.10 合成生成器 (generator.py)

主生成器协调整个合成流程。

#### SynthesisGenerator 类

```python
class SynthesisGenerator:
    """
    合成生成器

    主要职责：
    - 加载数据和配置
    - 协调任务执行
    - 管理检查点和输出
    """

    def __init__(self, config: SynthesisConfig):
        self.config = config

        # 加载 Data_Preparation 配置和数据集
        self.data_config = DataConfig()
        self.dataset = None  # 延迟加载

        # 初始化组件
        self.task_registry = TaskRegistry()
        self.api_client = OpenRouterClient(config)
        self.quality_filter = QualityFilter(config)
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir,
            config.checkpoint_interval
        )

        # k-NN 索引（延迟初始化）
        self.knn_index = None

        # 覆盖率追踪
        self.coverage = defaultdict(set)  # idx -> set(task_names)

    def _load_dataset(self, split: str):
        """加载数据集"""
        if self.dataset is None or self.dataset.split != split:
            self.dataset = ELMDataset(
                data_dir=self.data_config.data_dir,
                split=split,
                load_embeddings=True
            )

    def _build_knn_index(self):
        """构建 k-NN 索引"""
        if self.knn_index is None:
            self.knn_index = KNNIndex(
                self.dataset.embeddings,
                use_gpu=self.config.use_gpu_index
            )
```

#### 异步生成流程

```python
async def generate_for_split_async(self, split: str):
    """
    异步生成指定切分的合成数据

    流程：
    1. 加载数据集
    2. 构建 k-NN 索引
    3. 处理单文本任务（14 种）
    4. 处理配对任务（2 种）
    5. 输出统计信息
    """
    logger.info(f"开始处理 {split} 切分")

    # 加载数据
    self._load_dataset(split)
    self._build_knn_index()

    # 初始化输出写入器
    output_path = self.config.synthesis_dir / f'{split}_synthesis.jsonl'
    writer = OutputWriter(output_path)

    # 处理单文本任务
    single_tasks = self.task_registry.get_single_text_tasks()
    for task_config in single_tasks:
        await self._process_single_text_task_async(
            split, task_config, writer
        )

    # 处理配对任务
    pair_tasks = self.task_registry.get_pair_tasks()
    for task_config in pair_tasks:
        await self._process_pair_task_async(
            split, task_config, writer
        )

    writer.close()

    # 输出统计
    self._log_statistics(split)

async def _process_single_text_task_async(
    self,
    split: str,
    task_config: TaskConfig,
    writer: OutputWriter
):
    """
    异步处理单文本任务

    流程：
    1. 加载检查点（如果存在）
    2. 批量处理嵌入
    3. 并行调用 API
    4. 质量过滤
    5. 去重检查
    6. 写入输出
    7. 定期保存检查点
    """
    logger.info(f"处理任务: {task_config.name}")

    # 加载检查点
    checkpoint = self.checkpoint_manager.load_checkpoint(split, task_config.name)
    completed = checkpoint.completed_indices if checkpoint else set()
    total_gen = checkpoint.total_generations if checkpoint else 0
    rejection_count = checkpoint.rejection_count if checkpoint else 0

    # 获取待处理索引
    all_indices = list(range(len(self.dataset)))
    pending_indices = [i for i in all_indices if i not in completed]

    logger.info(f"待处理: {len(pending_indices)}/{len(all_indices)}")

    # 批量处理
    batch_semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)

    for batch_start in range(0, len(pending_indices), self.config.batch_size):
        batch_indices = pending_indices[batch_start:batch_start + self.config.batch_size]

        async with batch_semaphore:
            # 准备 prompts
            prompts = []
            for idx in batch_indices:
                sample = self.dataset[idx]
                prompt = task_config.prompt_template.format(text=sample['text'])
                prompts.append((idx, prompt, sample['text']))

            # 并行生成
            results = await self.api_client.generate_batch_async(
                [p[1] for p in prompts],
                task_config
            )

            # 处理结果
            for (idx, prompt, original_text), result in zip(prompts, results):
                if not result.success:
                    continue

                # 质量过滤
                passed, reason = self.quality_filter.filter(
                    result.text, original_text, task_config
                )

                if not passed:
                    rejection_count += 1
                    continue

                # 去重检查
                if await self.checkpoint_manager.is_duplicate_async(result.text):
                    continue

                # 写入输出
                output = SynthesisOutput(
                    task_type=task_config.name,
                    input_prompt_template=task_config.prompt_template,
                    embedding_index=idx,
                    target_text=result.text,
                    metadata={
                        'variation': 0,
                        'temperature': task_config.temperature,
                        'top_p': task_config.top_p,
                        'token_count': result.token_count
                    }
                )
                writer.write(output)

                # 更新追踪
                completed.add(idx)
                total_gen += 1
                self.coverage[idx].add(task_config.name)

        # 检查点
        if self.checkpoint_manager.should_checkpoint(total_gen):
            state = CheckpointState(
                split=split,
                task_name=task_config.name,
                completed_indices=completed,
                total_generations=total_gen,
                last_update=datetime.now().isoformat(),
                rejection_count=rejection_count,
                metadata={}
            )
            self.checkpoint_manager.save_checkpoint(state)

    logger.info(f"任务 {task_config.name} 完成: {total_gen} 生成, {rejection_count} 拒绝")
```

#### 配对任务处理

```python
async def _process_pair_task_async(
    self,
    split: str,
    task_config: TaskConfig,
    writer: OutputWriter
):
    """
    异步处理配对任务

    Compare 任务：
    - 对每个嵌入，找 k=3 个最近邻
    - 对每对（source, neighbor）生成比较

    Hypothetical 任务：
    - 对每个嵌入，找 k=2 个最近邻
    - 随机生成 alpha ∈ [0.3, 0.7]
    - 生成概念中点描述
    """
    logger.info(f"处理配对任务: {task_config.name}")

    # 加载检查点
    checkpoint = self.checkpoint_manager.load_checkpoint(split, task_config.name)
    completed = checkpoint.completed_indices if checkpoint else set()
    total_gen = checkpoint.total_generations if checkpoint else 0

    for idx in range(len(self.dataset)):
        if idx in completed:
            continue

        # 获取 k-NN 邻居
        neighbor_indices, similarities = self.knn_index.search(
            idx, task_config.k_neighbors
        )

        sample = self.dataset[idx]

        for neighbor_idx in neighbor_indices:
            neighbor_sample = self.dataset[neighbor_idx]

            # 构建 prompt
            if task_config.name == 'compare':
                prompt = task_config.prompt_template.format(
                    text1=sample['text'],
                    text2=neighbor_sample['text']
                )
                alpha = None
            else:  # hypothetical
                alpha = random.uniform(*task_config.alpha_range)
                prompt = task_config.prompt_template.format(
                    text1=sample['text'],
                    text2=neighbor_sample['text'],
                    alpha=alpha
                )

            # 生成
            result = await self.api_client.generate_async(prompt, task_config)

            if not result.success:
                continue

            # 质量过滤
            passed, _ = self.quality_filter.filter(
                result.text, sample['text'], task_config
            )

            if not passed:
                continue

            # 去重
            if await self.checkpoint_manager.is_duplicate_async(result.text):
                continue

            # 写入
            output = SynthesisOutput(
                task_type=task_config.name,
                input_prompt_template=task_config.prompt_template,
                embedding_index=idx,
                target_text=result.text,
                metadata={
                    'variation': 0,
                    'temperature': task_config.temperature,
                    'top_p': task_config.top_p,
                    'token_count': result.token_count,
                    'neighbor_idx': int(neighbor_idx),
                    'alpha': alpha
                }
            )
            writer.write(output)

            total_gen += 1
            self.coverage[idx].add(task_config.name)

        completed.add(idx)

        # 检查点
        if self.checkpoint_manager.should_checkpoint(total_gen):
            state = CheckpointState(
                split=split,
                task_name=task_config.name,
                completed_indices=completed,
                total_generations=total_gen,
                last_update=datetime.now().isoformat(),
                rejection_count=0,
                metadata={}
            )
            self.checkpoint_manager.save_checkpoint(state)
```

---

## 4. 第三阶段：模型训练 (Training)

模型训练阶段使用前两个阶段生成的数据训练 ELM MLP Adapter。

### 4.1 训练配置 (config.py)

训练配置模块定义了 Adapter 训练的所有参数：

#### TrainingConfig 数据类

```python
@dataclass
class TrainingConfig:
    """ELM Adapter 训练配置"""

    # 模型配置
    llm_model_name: str = "Qwen/Qwen3-4B-Instruct"  # 基础 LLM（冻结）
    embedding_dim: int = 2560                        # 嵌入维度
    hidden_dim: int = 4096                           # Adapter 中间层维度

    # 训练超参数
    learning_rate: float = 1e-4                      # 学习率
    warmup_steps: int = 1000                         # 预热步数
    weight_decay: float = 0.01                       # 权重衰减
    max_grad_norm: float = 1.0                       # 梯度裁剪

    # 批次配置（40GB VRAM 优化）
    batch_size: int = 16                             # 批次大小
    gradient_accumulation_steps: int = 2             # 梯度累积（有效批次=32）

    # 训练计划
    num_epochs: int = 3                              # 训练轮数
    eval_steps: int = 500                            # 评估间隔
    save_steps: int = 1000                           # 保存间隔

    # 内存优化
    use_bf16: bool = True                            # BFloat16 混合精度
    use_gradient_checkpointing: bool = True          # 梯度检查点

    # 特殊 token
    emb_token: str = "<EMB>"                         # 嵌入占位符 token
```

### 4.2 MLP Adapter 架构 (adapter.py)

#### EnhancedAdapter 设计

```python
class EnhancedAdapter(nn.Module):
    """
    增强型 MLP Adapter

    架构：
        输入: (batch, 2560)
        ↓
        Linear(2560 → 4096) + GELU
        ↓
        Linear(4096 → 2560)
        ↓
        Residual Connection: output = adapter(x) + x
        ↓
        LayerNorm(2560)
        ↓
        输出: (batch, 2560)

    参数量：~21M
        - Up projection: 2560 × 4096 + 4096 = 10.49M
        - Down projection: 4096 × 2560 + 2560 = 10.49M
        - LayerNorm: 2560 × 2 = 5K
    """

    def __init__(self, embedding_dim=2560, hidden_dim=4096):
        super().__init__()
        self.up_proj = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.GELU()
        self.down_proj = nn.Linear(hidden_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # 小标准差初始化（0.02）
        # 使得初始输出接近 0，让残差连接主导
        self._init_weights()

    def forward(self, x):
        residual = x
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        x = x + residual  # 残差连接
        x = self.layer_norm(x)
        return x
```

**为什么使用残差连接？**

- **稳定训练**：初期让 LLM 看到近似原始嵌入
- **梯度流动**：避免梯度消失
- **渐进学习**：Adapter 逐渐学习必要的变换

### 4.3 ELM 模型 (model.py)

#### 模型组件

```python
class ELMModel(nn.Module):
    """
    ELM 模型：冻结 LLM + 可训练 Adapter

    组件：
        E_0: 冻结的 token 嵌入层（Qwen3-4B-Instruct）
        E_A: 可训练的 MLP Adapter (~21M 参数)
        M_0: 冻结的 Transformer 层（Qwen3-4B-Instruct ~4B 参数）

    训练策略：
        ✓ 只训练 E_A
        ✗ E_0 和 M_0 完全冻结（requires_grad=False）
    """

    def __init__(self, config):
        super().__init__()

        # 1. 加载并配置 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<EMB>']})

        # 2. 加载冻结的 LLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # 3. 冻结所有 LLM 参数
        for param in self.llm.parameters():
            param.requires_grad = False

        # 4. 启用梯度检查点（节省显存）
        self.llm.gradient_checkpointing_enable()

        # 5. 创建可训练 Adapter
        self.adapter = EnhancedAdapter(config.embedding_dim, config.hidden_dim)
```

#### 前向传播流程

```python
def forward(self, input_ids, attention_mask, embeddings, embedding_positions, labels):
    """
    前向传播与嵌入注入

    步骤：
        1. E_0(input_ids) → token_embeds (batch, seq_len, 2560)
        2. E_A(embeddings) → adapted_embeds (batch, 2560)
        3. token_embeds[:, emb_pos] = adapted_embeds  # 注入
        4. M_0(token_embeds) → logits
        5. CrossEntropyLoss(logits, labels) → loss
    """
    # 获取 token 嵌入
    token_embeds = self.llm.model.embed_tokens(input_ids)

    # 通过 Adapter 变换外部嵌入
    adapted_embeds = self.adapter(embeddings)

    # 在 <EMB> 位置注入 adapted 嵌入
    for i, pos in enumerate(embedding_positions):
        token_embeds[i, pos] = adapted_embeds[i]

    # 通过冻结的 Transformer
    outputs = self.llm(inputs_embeds=token_embeds, attention_mask=attention_mask, labels=labels)

    return {'loss': outputs.loss, 'logits': outputs.logits}
```

### 4.4 训练数据集 (dataset.py)

#### ELMTrainingDataset

```python
class ELMTrainingDataset(Dataset):
    """
    ELM 训练数据集

    加载：
        - synthesis JSONL: task_type, embedding_index, target_text
        - embeddings SafeTensors: 预计算的嵌入向量

    返回：
        {
            'embedding': np.ndarray (2560,),
            'task_type': str,
            'target_text': str,
            'embedding_index': int
        }
    """

    def __init__(self, synthesis_path, embeddings_path):
        # 加载嵌入
        tensors = load_file(str(embeddings_path))
        self.embeddings = tensors["embeddings"]  # (N, 2560)

        # 加载 JSONL
        self.samples = []
        with open(synthesis_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __getitem__(self, idx):
        sample = self.samples[idx]
        embedding_idx = sample["embedding_index"]

        return {
            "embedding": self.embeddings[embedding_idx],
            "task_type": sample["task_type"],
            "target_text": sample["target_text"],
            "embedding_index": embedding_idx
        }
```

#### TrainingCollator

```python
class TrainingCollator:
    """
    训练批次整理器

    格式化输入：
        "<EMB> {task_prompt}\n{target_text}"

    任务 Prompt 映射：
        - keywords → "Extract key concepts:"
        - summary → "Summarize:"
        - compare → "Compare these:"
        ...（16 种任务类型）

    Label 掩码：
        - Prompt 部分：-100（不计算损失）
        - Target 部分：实际 token ID（计算损失）
    """

    TASK_PROMPTS = {
        "keywords": "Extract key concepts:",
        "summary": "Summarize:",
        "describe": "Describe in detail:",
        # ... 其他 13 种任务
    }

    def __call__(self, batch):
        # 为每个样本构建输入文本
        input_texts = []
        for sample in batch:
            prompt = self.TASK_PROMPTS[sample["task_type"]]
            input_text = f"<EMB> {prompt}\n{sample['target_text']}"
            input_texts.append(input_text)

        # Tokenize
        encodings = self.tokenizer(input_texts, padding=True, truncation=True, ...)

        # 创建 labels（掩码 prompt 部分）
        labels = encodings["input_ids"].clone()
        # ... 掩码逻辑

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "embeddings": torch.stack(embeddings),
            "embedding_positions": emb_positions,
            "labels": labels
        }
```

### 4.5 训练器 (trainer.py)

#### ELMTrainer 核心训练循环

```python
class ELMTrainer:
    """
    ELM Adapter 训练器

    功能：
        - 混合精度训练（BFloat16）
        - 梯度累积（有效批次=32）
        - 梯度裁剪（max_norm=1.0）
        - Warmup + 线性衰减调度
        - 定期评估和保存
    """

    def train_epoch(self, train_loader, val_loader, scaler):
        self.model.train()

        for step, batch in enumerate(train_loader):
            # 混合精度前向传播
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(**batch)
                loss = outputs['loss'] / self.config.gradient_accumulation_steps

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度累积步骤
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.adapter.parameters(),
                    self.config.max_grad_norm
                )

                # 优化器步骤
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # 定期评估
                if self.global_step % self.config.eval_steps == 0:
                    val_loss = self._evaluate(val_loader)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.checkpoint_manager.save_best(self.model.adapter, val_loss)
```

### 4.6 检查点管理 (checkpoint.py)

```python
class AdapterCheckpoint:
    """
    Adapter 检查点管理器

    保存策略：
        - 只保存 Adapter 权重（~21M 参数，而非完整 ~4B LLM）
        - SafeTensors 格式（adapter_step_X.safetensors）
        - 训练状态单独保存（checkpoint_step_X.pt）

    文件结构：
        data/checkpoints/
        ├── adapter_step_1000.safetensors  # Adapter 权重
        ├── checkpoint_step_1000.pt         # 优化器/调度器状态
        ├── adapter_best.safetensors        # 最佳模型
        └── best_model_meta.json            # 最佳模型元数据
    """

    def save(self, adapter, optimizer, scheduler, global_step, ...):
        # 保存 Adapter 权重
        adapter_path = f"adapter_step_{global_step}.safetensors"
        save_file(adapter.state_dict(), adapter_path)

        # 保存训练状态
        state_path = f"checkpoint_step_{global_step}.pt"
        torch.save({
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            ...
        }, state_path)
```

### 4.7 训练流程总结

```
数据加载：
  synthesis JSONL + embeddings SafeTensors
        ↓
  ELMTrainingDataset
        ↓
  TrainingCollator
        ↓
批次：{input_ids, attention_mask, embeddings, embedding_positions, labels}
        ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
训练循环（每个批次）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. E_0(input_ids) → token_embeds          [冻结]
  2. E_A(embeddings) → adapted_embeds       [可训练 ✓]
  3. 注入：token_embeds[:, emb_pos] = adapted_embeds
  4. M_0(token_embeds) → logits             [冻结]
  5. CrossEntropyLoss(logits, labels) → loss
        ↓
  6. loss.backward()  # 梯度只流向 E_A
  7. clip_grad_norm_(E_A.parameters(), 1.0)
  8. optimizer.step()  # 只更新 E_A
  9. scheduler.step()
        ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
每 500 步：验证集评估
每 1000 步：保存检查点
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 4.8 显存使用（40GB VRAM）

| 组件 | 显存占用 |
|------|---------|
| Qwen3-4B-Instruct (BF16) | ~8 GB |
| EnhancedAdapter | ~0.08 GB |
| 激活值 (batch=16, grad ckpt) | ~8 GB |
| 优化器状态 (仅 Adapter) | ~0.16 GB |
| **总计** | **~16-20 GB** |

**优化策略**：
- ✓ 梯度检查点：激活值显存减半
- ✓ 仅 Adapter 优化器状态
- ✓ BFloat16 混合精度
- ✗ 不需要 8-bit 量化（40GB 足够）

---

## 5. 数据流向图

### 5.1 完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ELM 数据流水线                                   │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────┐
                         │   WikiText-2     │
                         │  (HuggingFace)   │
                         └────────┬─────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     第一阶段：Data_Preparation                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  download.py │───▶│ preprocess.py│───▶│ embeddings.py│                  │
│  │              │    │              │    │              │                  │
│  │ 数据下载      │    │ 文本清洗      │    │ 嵌入生成      │                  │
│  │ 36K+ 行      │    │ 段落提取      │    │ Qwen3-4B     │                  │
│  │              │    │ Token 过滤    │    │ Last-token   │                  │
│  │              │    │ 数据切分      │    │ L2 归一化     │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                             │
│                              │                      │                       │
│                              ▼                      ▼                       │
│                     ┌──────────────┐       ┌──────────────┐                │
│                     │   Parquet    │       │ SafeTensors  │                │
│                     │ train.parquet│       │ train_emb.st │                │
│                     │  val.parquet │       │  val_emb.st  │                │
│                     │ test.parquet │       │ test_emb.st  │                │
│                     └──────────────┘       └──────────────┘                │
│                              │                      │                       │
└──────────────────────────────┼──────────────────────┼───────────────────────┘
                               │                      │
                               └──────────┬───────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     第二阶段：Data_Synthesis                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐                                                          │
│  │  ELMDataset  │◀───── 加载 Parquet + SafeTensors                         │
│  └──────┬───────┘                                                          │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐    ┌──────────────┐                                      │
│  │  KNN Index   │    │ Task Registry│                                      │
│  │   (FAISS)    │    │  (16 任务)   │                                      │
│  └──────┬───────┘    └──────┬───────┘                                      │
│         │                   │                                               │
│         └─────────┬─────────┘                                               │
│                   │                                                         │
│                   ▼                                                         │
│  ┌────────────────────────────────────────────────────────────────┐        │
│  │                      Generator                                  │        │
│  │  ┌──────────────────────────────────────────────────────────┐  │        │
│  │  │                  单文本任务 (14种)                         │  │        │
│  │  │  keywords, category, questions, summary, describe,       │  │        │
│  │  │  explain_beginner, explain_expert, related_topics,       │  │        │
│  │  │  characteristics_pos, characteristics_neg,               │  │        │
│  │  │  style_academic, style_casual, counterfactual            │  │        │
│  │  └──────────────────────────────────────────────────────────┘  │        │
│  │  ┌──────────────────────────────────────────────────────────┐  │        │
│  │  │                  配对任务 (2种)                           │  │        │
│  │  │  compare (k-NN 比较), hypothetical (嵌入插值)            │  │        │
│  │  └──────────────────────────────────────────────────────────┘  │        │
│  └────────────────────────────────────────────────────────────────┘        │
│                   │                                                         │
│                   ▼                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  API Client  │───▶│Quality Filter│───▶│ Checkpoint   │                  │
│  │  (OpenRouter)│    │  (5层检查)   │    │   Manager    │                  │
│  │Qwen3-30B-2507│    │              │    │ (断点续传)    │                  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                  │
│         │                                        │                         │
│         ├────────────────────────────────────────┤                         │
│         ▼                                        ▼                         │
│  ┌──────────────┐                       ┌──────────────┐                  │
│  │  Profiler    │                       │OutputWriter  │                  │
│  │ (性能追踪)    │                       │ (JSONL 写入) │                  │
│  └──────────────┘                       └──────┬───────┘                  │
│                                                 │                         │
│                                                 ▼                         │
│                                        ┌──────────────┐                  │
│                                        │    JSONL     │                  │
│                                        │ train_syn.jl │                  │
│                                        │  val_syn.jl  │                  │
│                                        │ test_syn.jl  │                  │
│                                        └──────┬───────┘                  │
│                                               │                           │
│                                               ▼                           │
│                                        ┌──────────────┐                  │
│                                        │  Validator   │                  │
│                                        │  (验证清单)   │                  │
│                                        └──────┬───────┘                  │
│                                               │                           │
└───────────────────────────────────────────────┼───────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     第三阶段：Training                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入数据：                                                                  │
│  ├── synthesis/train_synthesis.jsonl      （合成训练数据）                  │
│  ├── synthesis/val_synthesis.jsonl        （验证数据）                      │
│  ├── embeddings/train_embeddings.safetensors                                │
│  └── embeddings/val_embeddings.safetensors                                  │
│                   │                                                         │
│                   ▼                                                         │
│  ┌────────────────────────────────────────────────────────────────┐        │
│  │              ELMTrainingDataset + TrainingCollator              │        │
│  │  Format: "<EMB> {task_prompt}\n{target_text}"                  │        │
│  └─────────────────────────┬──────────────────────────────────────┘        │
│                            │                                                │
│                            ▼                                                │
│  ┌────────────────────────────────────────────────────────────────┐        │
│  │                      ELMModel                                   │        │
│  │  ┌──────────────────────────────────────────────────────────┐  │        │
│  │  │  E_0: Token Embeddings (Qwen3-4B-Instruct) [冻结]       │  │        │
│  │  │  E_A: MLP Adapter (~21M params)           [可训练 ✓]   │  │        │
│  │  │  M_0: Transformer Layers (~4B params)     [冻结]        │  │        │
│  │  └──────────────────────────────────────────────────────────┘  │        │
│  └─────────────────────────┬──────────────────────────────────────┘        │
│                            │                                                │
│                            ▼                                                │
│  ┌────────────────────────────────────────────────────────────────┐        │
│  │                    ELMTrainer                                   │        │
│  │  - BFloat16 混合精度                                            │        │
│  │  - 梯度累积（batch=16 × accum=2 = 有效批次 32）                │        │
│  │  - AdamW 优化器（仅 Adapter 参数）                             │        │
│  │  - Warmup + 线性衰减调度                                       │        │
│  │  - 梯度裁剪（max_norm=1.0）                                     │        │
│  └─────────────────────────┬──────────────────────────────────────┘        │
│                            │                                                │
│                            ▼                                                │
│  ┌──────────────┐    ┌──────────────┐                                      │
│  │AdapterCheckpoint  │  Evaluation   │                                      │
│  │  (每 1000 步)    │  (每 500 步)  │                                      │
│  └──────┬───────┘    └──────────────┘                                      │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐                                                          │
│  │ Checkpoints  │                                                          │
│  │adapter_best  │                                                          │
│  │.safetensors  │                                                          │
│  └──────────────┘                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 任务执行流程

```
对于每个嵌入 i：
├── 单文本任务 (14种)
│   ├── 构建 prompt: template.format(text=text_i)
│   ├── 调用 LLM (with rate limiting)
│   ├── 质量过滤 (5层)
│   ├── 去重检查 (MD5 hash)
│   └── 写入 JSONL
│
└── 配对任务 (2种)
    ├── 查找 k-NN 邻居: [j1, j2, ...]
    └── 对每个邻居 j：
        ├── Compare: prompt(text_i, text_j)
        ├── Hypothetical: prompt(text_i, text_j, alpha)
        ├── 调用 LLM
        ├── 质量过滤
        ├── 去重检查
        └── 写入 JSONL
```

---

## 6. 输出格式说明

### 6.1 Parquet 文件结构 (Data_Preparation 输出)

文件位置：`data/wikitext2_processed/{split}.parquet`

| 列名 | 类型 | 说明 |
|------|------|------|
| `text_id` | string | 唯一标识符（如 `train_0`） |
| `text` | string | 清洗后的文本 |
| `token_count` | int | Token 数量 |
| `char_count` | int | 字符数量 |

示例：
```
text_id      | text                                    | token_count | char_count
-------------|----------------------------------------|-------------|------------
train_0      | The tower is 324 metres tall...        | 156         | 892
train_1      | In 1889, the structure was...          | 203         | 1156
```

### 6.2 SafeTensors 嵌入格式 (Data_Preparation 输出)

文件位置：`data/embeddings/{split}_embeddings.safetensors`

| 键名 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `embeddings` | (n_samples, 2560) | float32 | L2 归一化嵌入 |

元数据：
```json
{
  "split": "train",
  "model": "Qwen/Qwen3-Embedding-4B",
  "num_texts": 4813,
  "embedding_dim": 2560
}
```

### 6.3 JSONL 合成数据格式 (Data_Synthesis 输出)

文件位置：`data/synthesis/{split}_synthesis.jsonl`

每行一个 JSON 对象：

#### 单文本任务输出

```json
{
  "task_type": "summary",
  "input_prompt_template": "Write a concise 2-3 sentence summary of the following text:\n\n{text}",
  "embedding_index": 42,
  "target_text": "This passage discusses the history of the Eiffel Tower...",
  "variation": 0,
  "temperature": 0.5,
  "top_p": 0.9,
  "token_count": 87
}
```

#### 配对任务输出 (Compare)

```json
{
  "task_type": "compare",
  "input_prompt_template": "Compare and contrast these two texts:\n\nText 1: {text1}\n\nText 2: {text2}",
  "embedding_index": 42,
  "target_text": "Both texts discuss architectural landmarks...",
  "variation": 0,
  "temperature": 0.5,
  "top_p": 0.9,
  "token_count": 156,
  "neighbor_idx": 87
}
```

#### 配对任务输出 (Hypothetical)

```json
{
  "task_type": "hypothetical",
  "input_prompt_template": "Describe a conceptual midpoint between these texts...",
  "embedding_index": 42,
  "target_text": "A hypothetical concept bridging these ideas would...",
  "variation": 0,
  "temperature": 0.7,
  "top_p": 0.92,
  "token_count": 134,
  "neighbor_idx": 87,
  "alpha": 0.45
}
```

### 6.4 Adapter 检查点格式 (Training 输出)

文件位置：`data/checkpoints/`

#### Adapter 权重文件

```
adapter_step_1000.safetensors    # 第 1000 步的 Adapter 权重
adapter_step_2000.safetensors    # 第 2000 步的 Adapter 权重
adapter_best.safetensors          # 验证集最佳模型
adapter_step_final.safetensors   # 最终模型
```

**文件大小**：~84 MB（21M 参数 × 4 字节/参数）

#### 训练状态文件

```
checkpoint_step_1000.pt           # 优化器/调度器状态
checkpoint_step_2000.pt
checkpoint_step_final.pt
```

包含内容：
```python
{
    "optimizer_state_dict": {...},    # AdamW 状态
    "scheduler_state_dict": {...},    # 学习率调度器状态
    "global_step": 1000,               # 全局步数
    "epoch": 0,                        # 当前轮次
    "best_val_loss": 2.34,             # 最佳验证损失
    "timestamp": "2025-12-12T10:30:00" # 保存时间
}
```

#### 最佳模型元数据

文件：`best_model_meta.json`

```json
{
    "val_loss": 2.234,
    "global_step": 1500,
    "epoch": 1,
    "timestamp": "2025-12-12T12:45:30"
}
```

---

## 7. 运行指南

### 7.1 环境配置

```bash
# 1. 克隆项目
cd /home/benz/coding_project/elm

# 2. 创建 Conda 环境
conda env create -f environment.yml

# 3. 激活环境
conda activate elm

# 4. 安装包（可选）
pip install -e .

# 5. 设置 API 密钥（Data_Synthesis 阶段需要）
export OPENROUTER_API_KEY="your-api-key"
```

### 7.2 运行数据准备阶段

```bash
# 完整流水线
python Data_Preparation/scripts/run_pipeline.py

# 常用参数
python Data_Preparation/scripts/run_pipeline.py \
    --batch-size 4 \          # 减小批次（显存不足时）
    --min-tokens 50 \          # 最小 token 数
    --max-tokens 1500 \        # 最大 token 数
    --no-flash-attention \     # 禁用 Flash Attention
    --skip-download \          # 跳过下载（已有数据）
    --skip-embeddings          # 跳过嵌入生成

# 仅重新生成嵌入
python Data_Preparation/scripts/run_embedding_only.py \
    --splits train val test \
    --batch-size 4
```

### 7.3 运行数据合成阶段

#### 基本用法

```bash
# 完整合成（所有切分：train, val, test）
python Data_Synthesis/scripts/run_synthesis.py

# 仅处理特定切分
python Data_Synthesis/scripts/run_synthesis.py --splits train
python Data_Synthesis/scripts/run_synthesis.py --splits train val
```

#### 参数调优

```bash
# 降低请求速率（避免 API 限制）
python Data_Synthesis/scripts/run_synthesis.py \
    --requests-per-second 5 \         # 默认 10 req/s
    --max-concurrent-requests 5       # 默认 10

# 调整批处理和并发
python Data_Synthesis/scripts/run_synthesis.py \
    --batch-size 30 \                 # 默认 50
    --max-concurrent-batches 2        # 默认 3

# 调整检查点间隔
python Data_Synthesis/scripts/run_synthesis.py \
    --checkpoint-interval 50          # 默认 20
```

#### 验证和工具

```bash
# 仅验证现有输出（不生成新数据）
python Data_Synthesis/scripts/run_synthesis.py --validate-only

# 排序 JSONL 文件
python scripts/sort_by_task_type.py

# 更新检查点（同步实际数据）
python scripts/update_checkpoint.py
```

#### 断点续传

检查点系统会自动保存进度，中断后重新运行相同命令即可恢复：

```bash
# 任务中断...
# Ctrl+C 或系统中断

# 重新运行，自动从检查点恢复
python Data_Synthesis/scripts/run_synthesis.py --splits train
```

### 7.4 运行模型训练阶段

#### 环境设置

```bash
# 1. 创建训练环境（与数据处理环境分离）
cd Training
conda env create -f environment.yml
conda activate elm-training

# 2. 验证数据文件存在
ls ../data/embeddings/train_embeddings.safetensors
ls ../data/embeddings/val_embeddings.safetensors
ls ../data/synthesis/train_synthesis.jsonl
ls ../data/synthesis/val_synthesis.jsonl
```

#### 基本训练

```bash
# 默认配置训练（推荐）
python scripts/train.py

# 完整参数示例
python scripts/train.py \
    --batch-size 16 \
    --grad-accum 2 \
    --epochs 3 \
    --lr 1e-4 \
    --warmup-steps 1000 \
    --hidden-dim 4096 \
    --eval-steps 500 \
    --save-steps 1000
```

#### 启用 W&B 日志

```bash
python scripts/train.py \
    --wandb \
    --wandb-project elm-adapter-training \
    --wandb-run-name exp-baseline
```

#### 从检查点恢复

```bash
# 自动从最新检查点恢复
python scripts/train.py --resume ../data/checkpoints/checkpoint_step_1000.pt

# 或指定 Adapter 文件
python scripts/train.py --resume ../data/checkpoints/adapter_step_1000.safetensors
```

#### 调整训练参数

```bash
# 更大的 Adapter（更多参数）
python scripts/train.py --hidden-dim 8192

# 更小的学习率
python scripts/train.py --lr 5e-5

# 更长的 warmup
python scripts/train.py --warmup-steps 2000

# 限制最大步数
python scripts/train.py --max-steps 5000
```

#### 评估已训练模型

```bash
# 在测试集上评估
python scripts/evaluate.py \
    --checkpoint ../data/checkpoints/adapter_best.safetensors \
    --split test \
    --num-samples 10
```

#### 预期训练时间

| 硬件 | 批次大小 | 有效批次 | 每步耗时 | 每轮耗时 | 3 轮总耗时 |
|------|---------|---------|---------|---------|-----------|
| A100 40GB | 16 | 32 | ~1.5s | ~1.5h | ~4.5h |
| A6000 48GB | 16 | 32 | ~2.0s | ~2.0h | ~6.0h |
| RTX 3090 24GB | 8 | 32 | ~1.8s | ~1.8h | ~5.4h |

（基于 train_synthesis.jsonl ~77,000 样本）

#### 监控训练

训练过程中监控的关键指标：

```
Step 500: val_loss = 2.456, best_val_loss = 2.456
Step 1000: val_loss = 2.234, best_val_loss = 2.234  ← 改进
Step 1500: val_loss = 2.189, best_val_loss = 2.189  ← 改进
Step 2000: val_loss = 2.201, best_val_loss = 2.189  ← 开始过拟合
```

**何时停止训练**：
- 验证损失连续 3-5 次评估不再下降
- 或达到目标轮数（默认 3 轮）

### 7.5 常见问题排查

#### CUDA 显存不足

```bash
# 解决方案 1：减小批次大小
python Data_Preparation/scripts/run_pipeline.py --batch-size 2

# 解决方案 2：禁用 FP16
python Data_Preparation/scripts/run_embedding_only.py --no-fp16

# 解决方案 3：使用 CPU（慢）
CUDA_VISIBLE_DEVICES="" python Data_Preparation/scripts/run_pipeline.py
```

#### Flash Attention 不可用

```bash
# 禁用 Flash Attention
python Data_Preparation/scripts/run_pipeline.py --no-flash-attention

# 或安装 Flash Attention
pip install flash-attn --no-build-isolation
```

#### API 速率限制

```bash
# 降低请求速率
python Data_Synthesis/scripts/run_synthesis.py --requests-per-second 0.3
```

#### 高拒绝率

检查日志中的拒绝原因，可能需要：
- 调整 `min_tokens` 阈值
- 修改 prompt 模板
- 检查质量过滤设置

#### 训练相关问题

##### 找不到合成数据文件

```bash
# 错误：FileNotFoundError: val_synthesis.jsonl not found
# 解决：生成验证集合成数据
cd ../Data_Synthesis
python scripts/run_synthesis.py --splits val
```

##### 显存不足（即使是 40GB）

```bash
# 解决方案 1：减小批次大小
python scripts/train.py --batch-size 8 --grad-accum 4

# 解决方案 2：禁用梯度检查点（更快但显存更多）
python scripts/train.py --no-grad-checkpoint

# 解决方案 3：使用更小的 Adapter
python scripts/train.py --hidden-dim 2048
```

##### 训练损失不下降

可能原因：
1. 学习率过高/过低 → 调整 `--lr`
2. Warmup 不足 → 增加 `--warmup-steps`
3. 数据质量问题 → 检查合成数据

```bash
# 尝试更保守的学习率
python scripts/train.py --lr 5e-5 --warmup-steps 2000
```

##### W&B 无法连接

```bash
# 离线模式训练（不记录到 W&B）
python scripts/train.py  # 不加 --wandb 参数
```

### 7.6 预期输出规模

| 阶段 | 输出 | 大小 |
|------|------|------|
| Data_Preparation | wikitext2_processed/ | ~4.5 MB |
| Data_Preparation | embeddings/ | ~59 MB |
| Data_Synthesis | synthesis/ | ~138 MB |

| 数据集 | 嵌入数量 | 预计合成样本 |
|--------|---------|-------------|
| train | 4,813 | ~231,000 |
| val | 601 | ~29,000 |
| test | 603 | ~29,000 |
| **总计** | **6,017** | **~289,000** |

| 检查点类型 | 数量 | 总大小 |
|-----------|------|-------|
| 训练中检查点 | ~10 个 | ~840 MB |
| 最佳模型 | 1 个 | ~84 MB |

---

## 8. 项目级工具脚本

### 8.1 按任务类型排序 (scripts/sort_by_task_type.py)

此脚本用于按任务类型对生成的 JSONL 文件进行排序，便于分析和组织数据。

#### 功能

```python
def sort_by_task_type(input_path: str, output_path: str | None = None):
    """
    按 task_type 字段排序 JSONL 文件

    参数：
        input_path: 输入 JSONL 文件路径
        output_path: 输出文件路径。如果为 None，则覆盖原文件

    功能：
    1. 读取所有记录
    2. 统计各任务类型数量
    3. 按 task_type 排序
    4. 写入排序后的记录
    """
```

#### 使用方法

```bash
# 默认排序 train_synthesis.jsonl（原地覆盖）
python scripts/sort_by_task_type.py

# 指定输入文件
python scripts/sort_by_task_type.py --input data/synthesis/train_synthesis.jsonl

# 指定输出文件（不覆盖原文件）
python scripts/sort_by_task_type.py \
    --input data/synthesis/train_synthesis.jsonl \
    --output data/synthesis/train_synthesis_sorted.jsonl
```

#### 输出示例

```
Loaded 77008 records

Task type counts:
  category: 4813
  characteristics_neg: 4813
  characteristics_pos: 4813
  compare: 14439
  counterfactual: 4813
  describe: 4813
  explain_beginner: 4813
  explain_expert: 4813
  hypothetical: 9626
  keywords: 4813
  questions: 4813
  related_topics: 4813
  style_academic: 4813
  style_casual: 4813
  summary: 4813

Sorted records written to: data/synthesis/train_synthesis.jsonl
```

### 8.2 更新检查点 (scripts/update_checkpoint.py)

此脚本用于根据实际生成的 JSONL 数据更新检查点文件，确保检查点状态与实际数据一致。

#### 功能

```python
def update_checkpoint(jsonl_path: Path, checkpoint_path: Path, task_name: str):
    """
    基于实际 JSONL 数据更新检查点文件

    参数：
        jsonl_path: train_synthesis.jsonl 路径
        checkpoint_path: 检查点文件路径
        task_name: 任务名称（如 "summary"）

    处理流程：
    1. 从 JSONL 读取指定任务的实际数据
    2. 提取已完成的嵌入索引
    3. 统计总生成数
    4. 更新检查点文件
    """
```

#### 使用方法

```bash
# 自动更新所有任务的检查点
python scripts/update_checkpoint.py
```

#### 工作原理

1. **自动发现任务类型**：扫描 `train_synthesis.jsonl` 发现所有唯一的 `task_type`
2. **逐任务更新**：为每个任务类型创建/更新对应的检查点文件
3. **准确性保证**：基于实际生成数据，而非推测值

#### 输出示例

```
============================================================
Discovering task types...
============================================================

Found 16 task types: category, characteristics_neg, characteristics_pos, compare, counterfactual, describe, explain_beginner, explain_expert, hypothetical, keywords, questions, related_topics, style_academic, style_casual, summary

============================================================
Processing task 1/16: category
============================================================
Reading data/synthesis/train_synthesis.jsonl...

Task: category
  Total records found: 4813
  Unique embedding indices: 4813
  Index range: 0 - 4812

Old checkpoint:
  Completed indices: 4500
  Total generations: 4500

Writing updated checkpoint to data/synthesis/checkpoints/train_category_checkpoint.json...
✅ Checkpoint updated successfully!

New checkpoint:
  Completed indices: 4813
  Total generations: 4813

[... 其他任务类似 ...]

============================================================
✅ All 16 checkpoints updated successfully!
============================================================
```

#### 使用场景

- **数据恢复**：手动修复或合并 JSONL 文件后，重新同步检查点
- **验证一致性**：确保检查点准确反映实际生成进度
- **调试**：当检查点与实际数据不匹配时，重建准确状态

---

## 9. 总结

ELM 项目是一个完整的数据处理和模型训练流水线，包含三个主要阶段：

1. **数据准备阶段**：从 WikiText-2 提取高质量文本，生成 Qwen3-Embedding-4B 嵌入向量
2. **数据合成阶段**：使用 Qwen3-30B-A3B-Instruct-2507 生成 16 种任务类型的多样化训练数据
3. **模型训练阶段**：训练 MLP Adapter 学习嵌入到 LLM token 空间的映射

### 关键技术特点

#### 数据准备阶段
- **高质量嵌入**：Last-token pooling + L2 归一化
- **高效存储**：Parquet（文本）+ SafeTensors（嵌入向量）
- **性能优化**：Flash Attention 2 + FP16 半精度
- **可配置过滤**：Token 数量、质量检查

#### 数据合成阶段
- **多样化任务**：事实/描述/创意/配对 四大类 16 种任务
- **k-NN 支持**：FAISS 索引实现高效相似度搜索
- **断点续传**：检查点系统支持长时间任务中断恢复
- **质量控制**：5 层过滤（最小 token、指令泄露、重复检测、无意义内容、相似度）+ MD5 去重
- **异步处理**：批量并发 API 调用，速率限制保护
- **性能分析**：实时追踪 API 延迟、吞吐量、拒绝率等指标
- **验证机制**：全面的输出质量检查清单

#### 模型训练阶段
- **参数高效训练**：仅训练 ~21M Adapter，冻结 ~4B LLM
- **混合精度**：BFloat16 自动混合精度训练
- **梯度优化**：梯度累积 + 梯度裁剪 + 梯度检查点
- **学习率调度**：Warmup + 线性衰减
- **检查点管理**：只保存轻量 Adapter 权重，支持断点续传
- **验证监控**：定期评估验证集，保存最佳模型
- **W&B 集成**：可选的实验追踪和可视化

#### 工具脚本
- **数据整理**：按任务类型排序 JSONL 文件
- **状态同步**：根据实际数据自动更新检查点

### 项目规模

| 组件 | 输出 | 说明 |
|------|------|------|
| Data_Preparation | 6,017 文本 + 嵌入 | ~59 MB（SafeTensors）|
| Data_Synthesis | ~289,000 合成样本 | ~138 MB（JSONL）|
| 任务类型 | 16 种 | 单文本 14 种 + 配对 2 种 |
| 质量保障 | 5 层过滤 + 去重 | 拒绝率 < 20% |
