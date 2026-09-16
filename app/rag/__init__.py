"""RAG 五层架构 —— 统一出口。

分层职责一览：

    L1 数据准备  prepare.py     清洗 / 去重 / 元数据注入
    L2 索引构建  indexer.py     切片 / 向量化 / 幂等入库
    L3 检索优化  retriever.py   多路召回 / RRF 融合 / 软回退
    L4 生成控制  generator.py   引用溯源 / 幻觉抑制 / 置信度拒答
    L5 评估迭代  evaluator.py   命中率 / 忠实度 / 反馈回流

分层带来的直接收益：
    - 每层可独立测试与替换（例如把 L3 换成 Elasticsearch 混合检索，L1/L2/L4 不动）；
    - 问题可归因（召回差是 L3 的锅，答非所问是 L4 的锅，语料脏是 L1 的锅）；
    - L5 的指标可以精确回流到对应层做定向调优，而不是全局瞎调参。
"""
from app.rag import evaluator, generator, indexer, prepare, retriever

# L1 数据准备
prepare_documents = prepare.prepare_documents
quality_report = prepare.quality_report

# L2 索引构建
build_index = indexer.build_index
add_document = indexer.add_document
delete_document = indexer.delete_document
chunk_documents = indexer.chunk_documents

# L3 检索优化
retrieve = retriever.retrieve
lexical_score = retriever.lexical_score
rrf_fuse = retriever.rrf_fuse

# L4 生成控制
generate_answer = generator.generate_answer
estimate_confidence = generator.estimate_confidence
build_context = generator.build_context

# L5 评估迭代
run_retrieval_eval = evaluator.run_retrieval_eval
score_faithfulness = evaluator.score_faithfulness
record_feedback = evaluator.record_feedback
build_report = evaluator.build_report
suggest_improvements = evaluator.suggest_improvements

__all__ = [
    "prepare",
    "indexer",
    "retriever",
    "generator",
    "evaluator",
    "prepare_documents",
    "quality_report",
    "build_index",
    "add_document",
    "delete_document",
    "chunk_documents",
    "retrieve",
    "lexical_score",
    "rrf_fuse",
    "generate_answer",
    "estimate_confidence",
    "build_context",
    "run_retrieval_eval",
    "score_faithfulness",
    "record_feedback",
    "build_report",
    "suggest_improvements",
]
