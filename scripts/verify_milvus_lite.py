"""在**真实 Milvus 引擎**上验证 MilvusVectorStore（无需 Docker）。

为什么需要它
------------
单测里用的是 fake 模块，只能证明「我自己的逻辑对」，证明不了「对接真实 API 对」。
实测被它抓到过一个 fake 永远抓不到的 bug：真实 ``MilvusClient.delete`` 返回的是
**被删主键列表**，而 fake 写成了 dict，于是「删除生效、返回 0」这个问题一直绿着。
凡是对接外部服务，都要有这一步真实校验。

Milvus Lite 是官方嵌入式版本，进程内运行、**不需要 Docker / etcd / minio**，
因此适合放进日常回归。它与 Standalone 共用同一套 API，差别只在索引实现与规模上限。
（HNSW 索引参数在本脚本里会自动回退 FLAT——Milvus Lite 对索引的支持与 Standalone
不同；HNSW 路径的正确性由真实库签名比对覆盖。）

怎么跑（真实库不要装进项目 venv，避免动到既有依赖）
---------------------------------------------------
    python -m venv /tmp/milvus-verify-venv
    /tmp/milvus-verify-venv/bin/pip install "pymilvus[milvus_lite]" numpy python-dotenv
    PYTHONPATH=. /tmp/milvus-verify-venv/bin/python scripts/verify_milvus_lite.py

本脚本依赖的 ``app/db/vector_db.py`` 只用到 numpy + dotenv + pymilvus，
所以隔离 venv 里装这三样即可跑**项目里那份真实适配器**，项目 venv 一行不动。

退出码：0 = 全部通过；1 = 有失败项。
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.vector_db import MilvusVectorStore

IDS = ["a#0", "b#0", "c#0"]
TEXTS = ["报销流程", "年假天数", "邮箱扩容"]
VECTORS = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]]
METAS = [
    {"source": "finance.md", "chunk": 0},
    {"source": "hr.md", "chunk": 0},
    {"source": "finance.md", "chunk": 1},
]

_checks = []


def check(label, condition, detail=""):
    _checks.append((label, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")


def run(index_type: str) -> None:
    dbdir = tempfile.mkdtemp(prefix="milvus_lite_")
    try:
        store = MilvusVectorStore(
            uri=os.path.join(dbdir, "kb.db"), collection_name="kb_verify", dim=0,
            index_type=index_type,
        )
        print(f"\n--- 真实引擎 roundtrip（index_type={index_type}）---")

        check("空库 count()==0", store.count() == 0)
        check("空库 search()==[]", store.search([1.0, 0.0, 0.0, 0.0], k=3) == [])
        check("空库 get_texts()==[]", store.get_texts() == [])
        check("空库 list_sources()==[]", store.list_sources() == [])
        check("空库 delete_by_source()==0", store.delete_by_source("nope.md") == 0)

        added = store.add(IDS, TEXTS, VECTORS, METAS)
        check("add() 返回 3", added == 3, f"实际 {added}")
        check("count()==3", store.count() == 3, f"实际 {store.count()}")

        hits = store.search([1.0, 0.0, 0.0, 0.0], k=2)
        got = [h.content for h in hits]
        check("search Top-2 命中正确", got == ["报销流程", "邮箱扩容"], f"实际 {got}")
        if len(hits) >= 2:
            check("COSINE 分数单调下降", hits[0].score >= hits[1].score,
                  f"{hits[0].score:.4f} >= {hits[1].score:.4f}")
            check("JSON 字段 meta 往返无损",
                  hits[0].metadata.get("source") == "finance.md", f"实际 {hits[0].metadata}")

        expected = [{"source": "finance.md", "chunks": 2}, {"source": "hr.md", "chunks": 1}]
        check("list_sources 聚合正确", store.list_sources() == expected,
              f"实际 {store.list_sources()}")
        check("get_texts 全量扫描正确", sorted(store.get_texts()) == sorted(TEXTS),
              f"实际 {store.get_texts()}")

        store.add(["a#0"], ["新文本"], [[1.0, 0.0, 0.0, 0.0]], [{"source": "finance.md"}])
        check("upsert 幂等（同 id 覆盖不追加）", store.count() == 3, f"实际 {store.count()}")
        check("upsert 内容已覆盖", "新文本" in store.get_texts())

        removed = store.delete_by_source("finance.md")
        check("delete_by_source 删除 2 行", removed == 2, f"实际 {removed}")
        check("删除后 count()==1", store.count() == 1, f"实际 {store.count()}")
        check("删除后 list_sources 只剩 hr.md",
              store.list_sources() == [{"source": "hr.md", "chunks": 1}])

        store.clear()
        check("clear() 后 count()==0", store.count() == 0)
        check("clear() 后 search()==[]", store.search([1.0, 0.0, 0.0, 0.0], k=3) == [])
    finally:
        shutil.rmtree(dbdir, ignore_errors=True)


def main() -> int:
    used = "HNSW"
    try:
        run("HNSW")
    except Exception as exc:  # noqa: BLE001
        print(f"\n!! HNSW 在 Milvus Lite 上不可用（{type(exc).__name__}: {exc}）")
        print("!! 回退 FLAT 重跑（HNSW 的正确性由真实库签名比对覆盖）")
        run("FLAT")
        used = "FLAT"

    passed = sum(1 for _, ok, _ in _checks if ok)
    failures = [c for c in _checks if not c[1]]
    print(f"\n=== 真实引擎验证结果（index_type={used}）：{passed}/{len(_checks)} 通过 ===")
    for label, _, detail in failures:
        print(f"  FAIL {label} {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
