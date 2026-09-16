"""切分改造 T2-2：结构感知切分独立验证。

验证点：
1. 默认 recursive 下，chunk_documents 行为与改造前一致（零变化由 test_chunking_baseline 兜底，此处只确认分发不误伤）。
2. CHUNK_STRATEGY=structure 下：
   - 跨章合并率显著低于 recursive（核心收益）；
   - 无结构文档（flat）自动回退递归，不报错；
   - 上下文头 / 元数据增强随各自 flag 开关生效、关闭即回退；
   - 索引膨胀不超过 2.0×。
"""
import re
import statistics as st

import pytest
from langchain_core.documents import Document

from app.rag import indexer
from app.rag.prepare import prepare_documents
from app.utils.doc_loader import load_all_documents

_CHAPTER_RE = re.compile(r"(第[一二三四五六七八九十百]+章|^\s*[一二三四五六七八九十]+、)", re.M)

#: 本文件按需改写的全局配置项（见下方 autouse fixture）
_CHUNK_ATTRS = (
    "CHUNK_STRATEGY", "CHUNK_CONTEXT_HEADER", "CHUNK_ENRICH_METADATA",
    "CHUNK_SIZE", "CHUNK_OVERLAP", "CHUNK_FALLBACK_OVERLAP_RATIO",
)


@pytest.fixture(autouse=True)
def _restore_chunk_config():
    """快照并在用例结束后还原切分配置，避免污染同进程内的其他测试文件。

    回归：本文件收尾处曾硬编码 `CHUNK_STRATEGY = "recursive"` —— 那是 T5-2 之前的
    默认值，与「代码默认与线上一致」的约定相悖。它会悄悄把 .env 的 structure 覆盖
    回 recursive，导致随后运行的 `test_chunking_baseline` 按错误策略比对而误报漂移，
    表现为「单跑绿、全量红」的顺序依赖。
    """
    originals = {name: getattr(indexer.config, name) for name in _CHUNK_ATTRS}
    yield
    for name, value in originals.items():
        setattr(indexer.config, name, value)


def _strip_header(text: str) -> str:
    if text.startswith("【章节】"):
        return text.split("\n\n", 1)[1] if "\n\n" in text else text
    return text


def _metrics(chunks):
    lens = [len(c.page_content) for c in chunks]
    cross = 0
    with_path = 0
    for c in chunks:
        body = _strip_header(c.page_content)
        if _CHAPTER_RE.search(body):
            if len(set(m.group(0) for m in _CHAPTER_RE.finditer(body))) >= 2:
                cross += 1
        if c.metadata.get("heading_path"):
            with_path += 1
    n = len(chunks)
    return {
        "count": n,
        "mean": st.mean(lens) if n else 0.0,
        "cross_chapter_rate": cross / n if n else 0.0,
        "with_heading_path": with_path,
    }


def _prep():
    return prepare_documents(load_all_documents())


def test_structure_reduces_cross_chapter():
    prep = _prep()
    # 递归基线
    indexer.config.CHUNK_STRATEGY = "recursive"
    rec = indexer.chunk_documents(prep)
    rec_m = _metrics(rec)

    # 结构策略（头/元数据全开）
    indexer.config.CHUNK_STRATEGY = "structure"
    indexer.config.CHUNK_CONTEXT_HEADER = True
    indexer.config.CHUNK_ENRICH_METADATA = True
    st_on = indexer.chunk_documents(prep)
    st_m = _metrics(st_on)

    print(f"[T2-2] recursive: count={rec_m['count']} cross={rec_m['cross_chapter_rate']:.0%}")
    print(f"[T2-2] structure : count={st_m['count']} cross={st_m['cross_chapter_rate']:.0%} "
          f"wMeta={st_m['with_heading_path']} inflation={st_m['count']/rec_m['count']:.2f}x")

    # 核心收益：跨章率必须显著下降（线性边界硬隔离）
    assert st_m["cross_chapter_rate"] < rec_m["cross_chapter_rate"], \
        f"结构策略跨章率未下降：{st_m['cross_chapter_rate']:.0%} >= {rec_m['cross_chapter_rate']:.0%}"
    # 膨胀受控
    assert st_m["count"] / rec_m["count"] <= 2.0, "索引膨胀超过 2.0×"
    # 元数据增强生效
    assert st_m["with_heading_path"] > 0, "结构策略未写入 heading_path 元数据"


def test_structure_header_and_metadata_toggle():
    prep = _prep()

    # 关闭头与元数据：无【章节】前缀、无 heading_path
    indexer.config.CHUNK_STRATEGY = "structure"
    indexer.config.CHUNK_CONTEXT_HEADER = False
    indexer.config.CHUNK_ENRICH_METADATA = False
    off = indexer.chunk_documents(prep)
    assert all(not c.page_content.startswith("【章节】") for c in off), "关闭头后仍出现【章节】前缀"
    assert all("heading_path" not in c.metadata for c in off), "关闭元数据后仍出现 heading_path"

    # 开启：应出现前缀与元数据
    indexer.config.CHUNK_CONTEXT_HEADER = True
    indexer.config.CHUNK_ENRICH_METADATA = True
    on = indexer.chunk_documents(prep)
    assert any(c.page_content.startswith("【章节】") for c in on), "开启头后未出现【章节】前缀"
    assert any(c.metadata.get("heading_path") for c in on), "开启元数据后未出现 heading_path"


def test_flat_doc_falls_back_to_recursive():
    """无结构文档（如速查表）走 structure 时应回退递归，不抛错且产出非空。"""
    prep = _prep()
    flat = [d for d in prep if "速查" in d.metadata.get("file_name", "")]
    assert flat, "测试语料应含扁平结构文档（系统权限与审批速查.txt）"

    indexer.config.CHUNK_STRATEGY = "structure"
    out = indexer.chunk_documents(flat)
    assert len(out) > 0, "扁平文档走 structure 未产出任何块"
    # 回退路径不写 heading_path（与 recursive 一致）
    assert all("heading_path" not in c.metadata for c in out), "扁平文档不应带 heading_path"


# ---------------------------------------------------------------------------
# C4 兼容层接线：CHUNK_FALLBACK_OVERLAP_RATIO
# ---------------------------------------------------------------------------
def _long_flat_doc() -> Document:
    """造一篇足够长、且无任何章节锚点的文档，逼出「重叠生效」的分割行为。"""
    text = "".join(f"第{i}行：用于测试降级路径重叠比例的中文文本内容。" for i in range(120))
    return Document(page_content=text, metadata={"source": "flat_demo.txt",
                                                 "file_name": "flat_demo.txt"})


def test_flat_fallback_overlap_ratio_is_consumed(monkeypatch):
    """降级路径的重叠比例必须由 CHUNK_FALLBACK_OVERLAP_RATIO 决定。

    回归：该配置曾「有声明、无读取点」——降级路径实际吃的是 `CHUNK_OVERLAP`，
    改这个值不会有任何行为变化，属于静默失效的伪配置。
    """
    doc = _long_flat_doc()
    monkeypatch.setattr(indexer.config, "CHUNK_OVERLAP", 60)

    monkeypatch.setattr(indexer.config, "CHUNK_FALLBACK_OVERLAP_RATIO", 0.12)
    small = [c.page_content for c in indexer._chunk_recursive(doc, degraded=True)]

    # 比例换算值等于 CHUNK_OVERLAP（0.2 × 300 = 60）时，降级路径与常规路径完全一致
    monkeypatch.setattr(indexer.config, "CHUNK_FALLBACK_OVERLAP_RATIO", 0.2)
    same = [c.page_content for c in indexer._chunk_recursive(doc, degraded=True)]
    normal = [c.page_content for c in indexer._chunk_recursive(doc, degraded=False)]

    assert small != normal, "改 CHUNK_FALLBACK_OVERLAP_RATIO 未产生任何差异，配置未被消费"
    assert same == normal, "比例设为 CHUNK_OVERLAP/CHUNK_SIZE 后应精确复现接线前行为"


def test_structure_flat_fallback_uses_degraded_overlap(monkeypatch):
    """structure 遇到无结构文档时，必须走「降级」分支而非普通递归分支。"""
    doc = _long_flat_doc()
    monkeypatch.setattr(indexer.config, "CHUNK_OVERLAP", 60)

    monkeypatch.setattr(indexer.config, "CHUNK_FALLBACK_OVERLAP_RATIO", 0.12)
    docs_small = indexer._chunk_structure(doc)

    monkeypatch.setattr(indexer.config, "CHUNK_FALLBACK_OVERLAP_RATIO", 0.2)
    docs_large = indexer._chunk_structure(doc)

    assert [c.page_content for c in docs_small] != [c.page_content for c in docs_large], \
        "structure 的 flat 回退未消费降级 overlap 比例"
    assert all("heading_path" not in c.metadata for c in docs_small), "flat 回退不应写 heading_path"
