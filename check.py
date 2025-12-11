import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

text = "Hello world, this is a test."

# 方法1: 你的实现
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B", padding_side='left')
model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-4B")
model.eval()

inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    emb1 = outputs.last_hidden_state[:, -1].numpy()  # last token

# 方法2: SentenceTransformers
st_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
emb2 = st_model.encode(text, normalize_embeddings=False)

# 比较
print(f"Max diff: {np.abs(emb1 - emb2).max()}")
print(f"Cosine sim: {np.dot(emb1.flatten(), emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))}")