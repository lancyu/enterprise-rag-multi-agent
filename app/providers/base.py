"""模型提供者抽象接口 —— 「模型相关」与「业务逻辑」之间的边界。

RAG 五层、图节点、API 只依赖这里的抽象接口（Embedder / Reranker / LangChain
BaseChatModel），不感知任何具体模型或供应商。换模型时只需改配置（.env）或新增
一个 provider 实现，业务代码零改动。

三类模型在标准 RAG 流程中的定位：
    - Embedder（向量化）：把文本变成归一化向量，供索引写入与 dense 检索；
    - BaseChatModel（生成/分类）：LLM 生成回答、意图分类、查询改写、记忆整理；
    - Reranker（精排，可选）：对召回候选做二次评分重排。
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from langchain_core.embeddings import Embeddings


class Embedder(Embeddings):
    """向量化接口：在 LangChain ``Embeddings`` 基础上增加「模式」与「维度」元信息。

    ``mode`` 用于「依赖 embedding 分数分布」的自适应逻辑（如检索阈值、软回退门槛）：
        - ``"api"``        真实神经向量（OpenAI 兼容 /embeddings 接口，语义最佳）
        - ``"local-hash"`` 本地零依赖哈希向量（无 Key 时的兜底）
    新增 provider（如 Ollama / 本地 ONNX）时，实现本接口并给 ``mode`` 取新值即可，
    无需改动检索层的阈值逻辑——它们只比较 ``mode == "local-hash"`` 这一个开关。
    """

    mode: str = "api"
    dim: int = 1024

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """把一批文档片段编码为归一化向量。"""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """把单条查询编码为归一化向量。"""


class Reranker(ABC):
    """精排接口：对召回候选做二次评分重排。

    默认实现走 sentence-transformers cross-encoder（见 app/providers/rerank.py）；
    也可替换为 API 精排服务 / 加权重排，只需实现本接口并在工厂里注册。
    """

    @abstractmethod
    def rerank(self, query: str, docs: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        """返回按相关性降序的候选列表（未启用或失败时原样返回，不丢片段）。"""
