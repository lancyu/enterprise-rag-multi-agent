#!/usr/bin/env python
"""向量库连接自检 —— 一条命令确认「配的后端能连上、能读能写」。

为什么需要它
------------
服务启动时若后端连不上，``get_vector_store()`` 会**降级为内存库**并继续运行。
这是有意为之（可用性与配置问题不互相掩盖），但代价是：
配置写的是 milvus、数据却写进了本地内存库，而日志只有一行 warning 很容易被忽略。
排查「到底连上没」不应该靠翻日志，所以这里给出一条独立、可反复执行的命令。

它做什么
--------
1. **配置解析**：打印归一化后的 ``VECTOR_DB_TYPE``、集合名、各后端参数（密钥打码）
2. **依赖体检**：``pymilvus`` / ``chromadb`` 是否已装，缺了直接告诉你要装什么
3. **连通性**：真实建立连接，顺带列出现有集合（确认连的是哪个实例）
4. **读写往返**：写入 → 计数 → 检索命中 → 列来源 → 按来源删除 → 计数归零
5. **清理**：探针数据落在独立的临时集合里，结束即删，**绝不触碰生产集合**

用法
----
    python scripts/check_vector_db.py                  # 按 .env 的 VECTOR_DB_TYPE 检查
    python scripts/check_vector_db.py --type milvus    # 临时改用 milvus（不改 .env）
    python scripts/check_vector_db.py --type chroma
    python scripts/check_vector_db.py --uri http://localhost:19530
    python scripts/check_vector_db.py --keep           # 保留探针集合，便于人工查看

退出码：0 = 全部通过；1 = 有失败项（可直接接进 CI / 启动脚本）。
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

# 允许直接以 `python scripts/xxx.py` 运行（scripts/ 不在包路径里）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.db.vector_db import (
    ChromaVectorStore,
    MemoryVectorStore,
    MilvusVectorStore,
    get_store_type,
    get_vector_store,
)

#: 探针集合名。必须同时满足两个后端的命名规则：
#:   Chroma：3-512 字符，仅 [a-zA-Z0-9._-]，且首尾必须是字母或数字（`_` 开头会被拒）
#:   Milvus：以字母或下划线开头，只允许字母/数字/下划线（连字符会被拒）
#: 因此取「字母开头 + 下划线分隔」这个交集，改名时务必两头都试。
PROBE_COLLECTION = "vector_conn_probe"
PROBE_SOURCE = "probe_source"
DIM = 8

_results: list = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    """记录一条检查结果；返回 condition 便于串接。"""
    _results.append((label, bool(condition), detail))
    flag = "PASS" if condition else "FAIL"
    line = f"  [{flag}] {label}"
    if detail:
        line += f" —— {detail}"
    print(line)
    return bool(condition)


def info(label: str, detail: str = "") -> None:
    """打印一条纯信息（不计入通过率）。"""
    print(f"  [INFO] {label}" + (f" —— {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def mask(secret: str) -> str:
    """密钥打码：只留长度信息，避免自检输出把 token 带进工单/日志。"""
    if not secret:
        return "(空)"
    return f"****（{len(secret)} 位）"


def _dep_installed(module: str) -> bool:
    """不 import 只探测：import 一个 C 扩展可能很慢或有副作用，find_spec 足够。"""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _probe_vectors():
    """构造可预测的探针向量：第 i 条为第 i 个基向量，查询 v0 时 Top-1 必为 doc0。"""
    vectors = []
    for i in range(3):
        vec = [0.0] * DIM
        vec[i] = 1.0
        vectors.append(vec)
    return vectors


def _cleanup_probe(store, backend: str, keep: bool) -> None:
    """删除探针集合。

    优先走适配器的 ``clear()``（接口内的公开方法）；Chroma 的 ``clear()`` 语义是
    「清空并重建」，会留下一个空集合，所以再补一次 best-effort 的整集合删除。
    清理失败不影响检查结论——它只是垃圾，不是正确性问题。
    """
    if keep:
        info("已保留探针集合（--keep）", PROBE_COLLECTION)
        return
    try:
        store.clear()
    except Exception as exc:  # noqa: BLE001
        info("clear() 未成功，尝试直接删除集合", str(exc)[:120])
    if backend == "chroma":
        try:
            store._client.delete_collection(PROBE_COLLECTION)
        except Exception:  # noqa: BLE001
            pass
    print(f"  探针集合已清理：{PROBE_COLLECTION}")


def _list_collections(store, backend: str) -> None:
    """列出现有集合（只读、best-effort）：用于确认「连的到底是哪个实例」。"""
    try:
        if backend == "milvus":
            names = store._client.list_collections()
        elif backend == "chroma":
            names = [c.name for c in store._client.list_collections()]
        else:
            info("内存库无「集合」概念，跳过")
            return
        shown = [n for n in names if n != PROBE_COLLECTION]
        info(f"该实例现有集合 {len(shown)} 个", ", ".join(map(str, shown[:10])) or "(无)")
    except Exception as exc:  # noqa: BLE001
        info("列举集合失败（不影响连通性结论）", str(exc)[:120])


def run(backend: str, keep: bool, args) -> int:
    print("=" * 72)
    print("向量库连接自检")
    print("=" * 72)

    # ---------- 1. 配置解析 ----------
    section("1. 配置解析")
    info("VECTOR_DB_TYPE", f"{config.VECTOR_DB_TYPE}（本次检查用：{backend}）")
    if backend != config.VECTOR_DB_TYPE:
        info("注意", "--type 覆盖了 .env 里的配置，本次不会改动任何配置文件")
    info("集合名", config.MILVUS_COLLECTION if backend == "milvus" else config.COLLECTION_NAME)
    if backend == "milvus":
        info("MILVUS_URI", args.uri or config.MILVUS_URI)
        info("MILVUS_TOKEN", mask(args.token if args.token is not None else config.MILVUS_TOKEN))
        info("MILVUS_DIM", str(config.MILVUS_DIM) if config.MILVUS_DIM else "0（按首次写入自动推断）")
        info("索引 / 距离", f"{config.MILVUS_INDEX_TYPE} / {config.MILVUS_METRIC_TYPE}")
        info("严格模式", "开（连不上即拒绝启动）" if config.VECTOR_DB_STRICT else "关（连不上则降级为内存库）")
    elif backend == "chroma":
        info("CHROMA_PERSIST_DIR", config.CHROMA_PERSIST_DIR)

    # ---------- 2. 依赖体检 ----------
    section("2. 客户端依赖")
    if backend == "milvus":
        check("pymilvus 已安装", _dep_installed("pymilvus"),
              "缺失请执行 pip install -r requirements-vector.txt")
    elif backend == "chroma":
        check("chromadb 已安装", _dep_installed("chromadb"),
              "缺失请执行 pip install -r requirements-vector.txt")
    else:
        info("内存库为零依赖实现，无需客户端包")

    # ---------- 3. 建立连接 ----------
    section("3. 建立连接")
    tmp_dir = None
    if backend == "milvus":
        constructor = lambda: MilvusVectorStore(
            uri=args.uri or config.MILVUS_URI,
            collection_name=PROBE_COLLECTION,
            token=args.token if args.token is not None else config.MILVUS_TOKEN,
            dim=0,  # 0 = 按首次写入的向量长度推断，避免维度写错时误判
            index_type=config.MILVUS_INDEX_TYPE,
            metric_type=config.MILVUS_METRIC_TYPE,
        )
    elif backend == "chroma":
        # 探针写进临时目录：不往用户的 vector_store/ 里塞垃圾
        tmp_dir = tempfile.mkdtemp(prefix="chroma-probe-")
        constructor = lambda: ChromaVectorStore(Path(tmp_dir), PROBE_COLLECTION)
    else:
        tmp_dir = tempfile.mkdtemp(prefix="memory-probe-")
        constructor = lambda: MemoryVectorStore(Path(tmp_dir))

    try:
        store = constructor()
    except Exception as exc:  # noqa: BLE001
        check("连接建立", False, f"{type(exc).__name__}: {str(exc)[:200]}")
        print("\n提示：后端服务未启动或地址/端口不对。")
        if backend == "milvus":
            print("      docker compose --profile milvus up -d   # 起 milvus + etcd + minio")
            print(f"      确认 {args.uri or config.MILVUS_URI} 可达（本地默认 19530 端口）")
        _summary(backend)
        return 1

    check("连接建立", True, type(store).__name__)
    _list_collections(store, backend)

    # ---------- 4. 读写往返 ----------
    section("4. 读写往返验证")
    vectors = _probe_vectors()
    ids = [f"{PROBE_SOURCE}#{i}" for i in range(3)]
    texts = [f"探针文本 {i}" for i in range(3)]
    metas = [{"source": PROBE_SOURCE, "idx": i} for i in range(3)]

    try:
        added = store.add(ids, texts, vectors, metas)
        check("写入 add()", added == 3, f"返回 {added}")

        count = store.count()
        check("计数 count()", count == 3, f"返回 {count}")

        hits = store.search(vectors[0], k=3)
        check("检索 search() 有结果", len(hits) > 0, f"返回 {len(hits)} 条")
        if hits:
            check("检索 Top-1 命中同向量文档", hits[0].content == texts[0],
                  f"Top-1={hits[0].content!r} score={hits[0].score:.4f}")
            # 三种后端的分数都归一化到余弦相似度（同向量应为 1.0 附近）
            check("相似度量纲一致（≈1.0）", hits[0].score > 0.9, f"score={hits[0].score:.4f}")
            check("元数据回传", hits[0].metadata.get("idx") == 0,
                  f"metadata={hits[0].metadata}")

        sources = store.list_sources()
        hit_src = [s for s in sources if s["source"] == PROBE_SOURCE]
        check("来源聚合 list_sources()", bool(hit_src) and hit_src[0]["chunks"] == 3,
              f"{hit_src[:1]}")

        texts_back = store.get_texts()
        check("全量回捞 get_texts()", texts[0] in texts_back, f"共 {len(texts_back)} 条")

        removed = store.delete_by_source(PROBE_SOURCE)
        check("按来源删除 delete_by_source()", removed == 3, f"返回 {removed}")

        left = store.count()
        check("删除后计数归零", left == 0, f"剩余 {left}")
    except Exception as exc:  # noqa: BLE001
        check("读写往返", False, f"{type(exc).__name__}: {str(exc)[:200]}")
        traceback.print_exc()
    finally:
        try:
            _cleanup_probe(store, backend, keep)
        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---------- 5. 工厂路径 ----------
    section("5. 工厂路径（服务启动时真正走的代码）")
    try:
        get_vector_store()
        effective = get_store_type()
        requested = config.VECTOR_DB_TYPE
        check("工厂成功返回实例", True, f"实际生效：{effective}")
        check("实际生效 == 配置要求", effective == requested,
              f"要求 {requested}，实际 {effective}")
        if effective != requested:
            info("发生了降级", "数据会写入本地内存库；如需拒绝启动请设 VECTOR_DB_STRICT=true")
    except Exception as exc:  # noqa: BLE001
        check("工厂成功返回实例", False, f"{type(exc).__name__}: {str(exc)[:200]}")

    return _summary(backend)


def _summary(backend: str) -> int:
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print("\n" + "=" * 72)
    if passed == total and total:
        print(f"✅ 全部通过：{passed}/{total}（后端 {backend} 可正常读写）")
        return 0
    print(f"❌ 未通过：{passed}/{total}")
    for label, ok, detail in _results:
        if not ok:
            print(f"   ✗ {label}" + (f" —— {detail}" if detail else ""))
    print("=" * 72)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="向量库连接自检（不触碰生产集合）")
    parser.add_argument("--type", choices=list(config.VECTOR_DB_CHOICES),
                        help="临时指定后端，默认沿用 .env 的 VECTOR_DB_TYPE")
    parser.add_argument("--uri", help="覆盖 Milvus 地址，如 http://localhost:19530")
    parser.add_argument("--token", help="覆盖 Milvus token（本地免鉴权可留空）")
    parser.add_argument("--keep", action="store_true", help="保留探针集合，便于人工查看")
    args = parser.parse_args()

    backend = args.type or config.VECTOR_DB_TYPE
    if backend not in config.VECTOR_DB_CHOICES:
        print(f"❌ VECTOR_DB_TYPE={backend!r} 不是合法后端；"
              f"可选值：{' | '.join(config.VECTOR_DB_CHOICES)}")
        return 1
    return run(backend, args.keep, args)


if __name__ == "__main__":
    sys.exit(main())
