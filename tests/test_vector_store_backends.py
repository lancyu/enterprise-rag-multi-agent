"""向量库三后端的一致性测试。

三个后端（memory / chroma / milvus）是**鸭子类型**接口，没有 ABC/Protocol 兜底。
漏实现一个方法不会在 import 期报错，只会在跑到那条路径时才炸——所以这里用
两组测试把它钉死：

1. **接口完整性**：三个类都必须实现全部 7 个方法（只看类属性，零依赖不实例化）；
2. **Milvus 适配器逻辑**：注入一个 fake ``pymilvus`` 模块，在没有真实服务的情况下
   验证 add / search / delete_by_source / count / clear / list_sources / get_texts
   的语义与内存库一致。这样 CI 不必起 Milvus 三容器，也能守住适配器不漂移。

3. **配置解析与降级策略**：``VECTOR_DB_TYPE`` 的归一化与非法值拒绝、
   ``VECTOR_DB_STRICT`` 的快速失败——这三条决定了「配错后端」时会怎样失败。

真实 Milvus 服务端的连通性**不在单元测试范围内**（CI 不起三容器），改由两个脚本承担：
不需要起服务的真实引擎往返验证见 `scripts/verify_milvus_lite.py`（Milvus Lite 嵌入式引擎），
面向使用者的连接自检见 `scripts/check_vector_db.py`。
"""
import math
import sys
import types

import pytest

from app.db.vector_db import (
    ChromaVectorStore,
    MemoryVectorStore,
    MilvusVectorStore,
)

REQUIRED_METHODS = (
    "add",
    "search",
    "delete_by_source",
    "count",
    "clear",
    "list_sources",
    "get_texts",
)


# ---------------------------------------------------------------------------
# 1. 接口完整性（零依赖，不实例化）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "store_cls", [MemoryVectorStore, ChromaVectorStore, MilvusVectorStore]
)
def test_backend_exposes_full_interface(store_cls):
    missing = [m for m in REQUIRED_METHODS if not callable(getattr(store_cls, m, None))]
    assert not missing, f"{store_cls.__name__} 缺少接口方法：{missing}"


# ---------------------------------------------------------------------------
# 2. fake pymilvus：不依赖真实服务的适配器测试
# ---------------------------------------------------------------------------
class _FakeSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, field_name, datatype, **kwargs):
        self.fields.append({"field_name": field_name, "datatype": datatype, **kwargs})


class _FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, field_name, index_type="", metric_type="", params=None):
        self.indexes.append(
            {
                "field_name": field_name,
                "index_type": index_type,
                "metric_type": metric_type,
                "params": params or {},
            }
        )


class _FakeDataType:
    VARCHAR = "VARCHAR"
    FLOAT_VECTOR = "FLOAT_VECTOR"
    JSON = "JSON"


class _FakeMilvusClient:
    """够用的 Milvus 内存替身：只实现适配器真正会调到的那部分。"""

    def __init__(self, uri=None, token=None):
        self.uri = uri
        self.token = token
        self.collections = {}
        self.dims = {}
        self.upserts = 0

    def create_schema(self, auto_id=False, enable_dynamic_field=False):
        return _FakeSchema()

    def prepare_index_params(self):
        return _FakeIndexParams()

    def has_collection(self, name):
        return name in self.collections

    def create_collection(self, collection_name, schema=None, index_params=None):
        dim = next((f["dim"] for f in schema.fields if f["datatype"] == "FLOAT_VECTOR"), 0)
        self.collections[collection_name] = []
        self.dims[collection_name] = dim

    def load_collection(self, name):
        if name not in self.collections:
            raise RuntimeError(f"collection {name} not found")

    def drop_collection(self, name):
        self.collections.pop(name, None)
        self.dims.pop(name, None)

    def upsert(self, collection_name, data):
        rows = self.collections[collection_name]
        index = {r["id"]: i for i, r in enumerate(rows)}
        for row in data:
            if row["id"] in index:
                rows[index[row["id"]]] = row
            else:
                index[row["id"]] = len(rows)
                rows.append(row)
        self.upserts += 1
        return {"upsert_count": len(data)}

    def delete(self, collection_name, filter):
        prefix, suffix = 'source == "', '"'
        assert filter.startswith(prefix) and filter.endswith(suffix), filter
        source = filter[len(prefix):-len(suffix)]
        rows = self.collections[collection_name]
        removed = [r["id"] for r in rows if r["source"] == source]
        self.collections[collection_name] = [r for r in rows if r["source"] != source]
        # 与真实 MilvusClient 一致：返回被删除的主键列表（不是在真实引擎上
        # 实测过之前，这里写成了 dict，正好把 delete_by_source 的读取 bug 掩盖掉）。
        return removed

    def search(self, collection_name, data, limit, output_fields, search_params):
        query = data[0]
        rows = self.collections[collection_name]
        scored = sorted(rows, key=lambda r: -_cosine(query, r["vector"]))[:limit]
        return [
            [
                {
                    "id": r["id"],
                    "distance": _cosine(query, r["vector"]),
                    "entity": {"text": r["text"], "source": r["source"], "meta": r["meta"]},
                }
                for r in scored
            ]
        ]

    def query(self, collection_name, filter, output_fields, limit, offset):
        rows = self.collections[collection_name]
        return [{k: r[k] for k in output_fields} for r in rows[offset:offset + limit]]

    def get_collection_stats(self, collection_name):
        return {"row_count": len(self.collections[collection_name])}


def _cosine(a, b):
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


@pytest.fixture
def fake_milvus(monkeypatch):
    module = types.ModuleType("pymilvus")
    module.MilvusClient = _FakeMilvusClient
    module.DataType = _FakeDataType
    monkeypatch.setitem(sys.modules, "pymilvus", module)
    return module


@pytest.fixture
def milvus_store(fake_milvus):
    return MilvusVectorStore(uri="http://fake:19530", collection_name="kb_test", dim=0)


def test_milvus_empty_before_first_write(milvus_store):
    """集合惰性创建：没写过任何数据时，读操作必须返回空而不是报错。"""
    assert milvus_store.count() == 0
    assert milvus_store.search([1.0, 0.0, 0.0, 0.0], k=3) == []
    assert milvus_store.get_texts() == []
    assert milvus_store.list_sources() == []
    assert milvus_store.delete_by_source("nope.md") == 0


def test_milvus_add_search_roundtrip(milvus_store):
    ids = ["a#0", "b#0", "c#0"]
    texts = ["报销流程", "年假天数", "邮箱扩容"]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]]
    metas = [
        {"source": "finance.md", "chunk": 0},
        {"source": "hr.md", "chunk": 0},
        {"source": "finance.md", "chunk": 1},
    ]

    assert milvus_store.add(ids, texts, vectors, metas) == 3
    assert milvus_store.count() == 3

    hits = milvus_store.search([1.0, 0.0, 0.0, 0.0], k=2)
    assert [h.content for h in hits] == ["报销流程", "邮箱扩容"]
    assert hits[0].score > hits[1].score
    assert hits[0].metadata["source"] == "finance.md"


def test_milvus_upsert_is_idempotent(milvus_store):
    """同 id 重复写入应当覆盖而不是追加（与内存库「先删后增」语义一致）。"""
    milvus_store.add(["a#0"], ["旧文本"], [[1.0, 0.0, 0.0, 0.0]], [{"source": "f.md"}])
    milvus_store.add(["a#0"], ["新文本"], [[1.0, 0.0, 0.0, 0.0]], [{"source": "f.md"}])

    assert milvus_store.count() == 1
    assert milvus_store.get_texts() == ["新文本"]


def test_milvus_delete_by_source_and_list_sources(milvus_store):
    milvus_store.add(
        ["a#0", "a#1", "b#0"],
        ["A0", "A1", "B0"],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [{"source": "a.md"}, {"source": "a.md"}, {"source": "b.md"}],
    )

    assert milvus_store.list_sources() == [
        {"source": "a.md", "chunks": 2},
        {"source": "b.md", "chunks": 1},
    ]
    assert milvus_store.delete_by_source("a.md") == 2
    assert milvus_store.count() == 1
    assert milvus_store.list_sources() == [{"source": "b.md", "chunks": 1}]


def test_milvus_delete_reads_all_return_shapes(milvus_store, monkeypatch):
    """delete 的返回值形态随 pymilvus 版本 / 后端而异，三种都必须读出删除数。

    真实引擎（Milvus Lite + pymilvus 2.6）返回 list；只认 dict 会让
    「删除确实生效、返回 0」——这个 bug 只有跑真实引擎才会暴露。
    """

    class _MutationResult:
        delete_count = 5

    milvus_store.add(["a#0"], ["A"], [[1.0, 0.0, 0.0, 0.0]], [{"source": "a.md"}])

    monkeypatch.setattr(milvus_store._client, "delete", lambda **kw: ["x#0", "x#1"])
    assert milvus_store.delete_by_source("a.md") == 2

    monkeypatch.setattr(milvus_store._client, "delete", lambda **kw: {"delete_count": 7})
    assert milvus_store.delete_by_source("a.md") == 7

    monkeypatch.setattr(milvus_store._client, "delete", lambda **kw: _MutationResult())
    assert milvus_store.delete_by_source("a.md") == 5


def test_milvus_clear_drops_collection(milvus_store):
    milvus_store.add(["a#0"], ["A"], [[1.0, 0.0]], [{"source": "a.md"}])
    assert milvus_store.count() == 1

    milvus_store.clear()

    assert milvus_store.count() == 0
    assert milvus_store.search([1.0, 0.0], k=1) == []


def test_milvus_dimension_mismatch_is_rejected(milvus_store):
    """集合已按 4 维建好后再写 3 维，必须报错而不是静默写坏。"""
    milvus_store.add(["a"], ["A"], [[1.0, 0.0, 0.0, 0.0]], [{"source": "a.md"}])

    with pytest.raises(ValueError, match="维度"):
        milvus_store.add(["b"], ["B"], [[1.0, 0.0, 0.0]], [{"source": "b.md"}])


# ---------------------------------------------------------------------------
# 3. 工厂：后端不可用时降级，而不是让服务起不来
# ---------------------------------------------------------------------------
def test_factory_falls_back_to_memory_when_milvus_missing(monkeypatch, tmp_path):
    import app.db.vector_db as vector_db

    monkeypatch.setattr(vector_db.config, "VECTOR_DB_TYPE", "milvus")
    monkeypatch.setattr(vector_db.config, "CHROMA_PERSIST_DIR", str(tmp_path))
    monkeypatch.setattr(vector_db, "_store_instance", None)
    monkeypatch.setattr(vector_db, "_store_type", None)
    # sys.modules 里放 None → `import pymilvus` 抛 ImportError
    monkeypatch.setitem(sys.modules, "pymilvus", None)

    store = vector_db.get_vector_store()

    assert isinstance(store, MemoryVectorStore)
    assert vector_db.get_store_type() == "memory"


def test_config_snapshot_reports_effective_collection(monkeypatch):
    """配置快照要报**当前后端生效**的集合名。

    Milvus 可用 MILVUS_COLLECTION 单独指定集合，若快照永远报 COLLECTION_NAME，
    运维在 /health 上看到的会是「另一个后端的集合名」，排查时被带偏。
    """
    from app import config as app_config

    monkeypatch.setattr(app_config, "VECTOR_DB_TYPE", "memory")
    monkeypatch.setattr(app_config, "COLLECTION_NAME", "kb_default")
    assert app_config.effective_vector_collection() == "kb_default"

    monkeypatch.setattr(app_config, "VECTOR_DB_TYPE", "milvus")
    monkeypatch.setattr(app_config, "MILVUS_COLLECTION", "kb_milvus")
    assert app_config.effective_vector_collection() == "kb_milvus"
    assert app_config.dump_config()["vector_db"]["collection"] == "kb_milvus"


# ---------------------------------------------------------------------------
# 4. 配置解析：写错要立刻报错，而不是静默退化成内存库
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["memory", "chroma", "milvus"])
def test_validate_vector_db_type_accepts_choices(monkeypatch, value):
    from app import config as app_config

    monkeypatch.setattr(app_config, "VECTOR_DB_TYPE", value)
    assert app_config.validate_vector_db_type() == value


def test_factory_rejects_unknown_backend(monkeypatch):
    """VECTOR_DB_TYPE 拼错（如 mlivus）必须抛错。

    过去它会被静默当成内存库：服务照常启动、数据写进本地文件，
    「配置写错了」与「本来就配内存库」表现完全一致，排查时毫无线索。
    """
    import app.db.vector_db as vector_db

    monkeypatch.setattr(vector_db.config, "VECTOR_DB_TYPE", "mlivus")
    monkeypatch.setattr(vector_db, "_store_instance", None)
    monkeypatch.setattr(vector_db, "_store_type", None)

    with pytest.raises(ValueError, match="mlivus"):
        vector_db.get_vector_store()


def test_factory_fails_fast_in_strict_mode(monkeypatch, tmp_path):
    """VECTOR_DB_STRICT=true 时，配了 milvus 却连不上必须拒绝启动。

    默认的「降级」保住了可用性，但代价是写入悄悄落到内存库。
    严格模式把问题挡在写入之前——两种取舍都要有，由用户选。
    """
    import app.db.vector_db as vector_db

    monkeypatch.setattr(vector_db.config, "VECTOR_DB_TYPE", "milvus")
    monkeypatch.setattr(vector_db.config, "VECTOR_DB_STRICT", True)
    monkeypatch.setattr(vector_db.config, "CHROMA_PERSIST_DIR", str(tmp_path))
    monkeypatch.setattr(vector_db, "_store_instance", None)
    monkeypatch.setattr(vector_db, "_store_type", None)
    monkeypatch.setitem(sys.modules, "pymilvus", None)

    with pytest.raises(RuntimeError, match="VECTOR_DB_STRICT"):
        vector_db.get_vector_store()


def test_vector_db_type_is_normalized_from_env():
    """`.env` 里写 ``" Milvus "`` 也要被识别成 ``milvus``。

    归一化发生在 config 导入期，只能在子进程里真跑一遍才测得到。
    不做归一化的话，大小写或空格写错会静默落进内存库分支。
    """
    import os
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    code = "import app.config as c; print(c.VECTOR_DB_TYPE); print(c.validate_vector_db_type())"
    env = {**os.environ, "VECTOR_DB_TYPE": "  Milvus  "}

    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(root), env=env, timeout=120,
    )

    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["milvus", "milvus"]

