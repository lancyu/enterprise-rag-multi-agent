"""全局配置中心 —— 统一管理模型、向量库、缓存与检索参数。

所有配置均支持环境变量覆盖，无需修改核心代码即可适配开发 / 生产环境。
"""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    """读取环境变量；值为空字符串或纯空白时回落到默认值。

    解决 .env 中 ``KEY=``（空值）被当成 ``''`` 而引发 ``int()/float()`` 崩溃的问题。
    """
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    return val


def _env_bool(key: str, default: bool) -> bool:
    """读取布尔型环境变量。**全项目布尔配置的唯一解析处。**

    本函数当初就是为了统一 ``os.getenv(k, "false").strip().lower() not in (...)``
    这种写法而建的，但建完之后**仍有 8 处**在用内联写法，且两边的判定集合不同：
    内联版不认 ``"none"``，于是 ``MEMORY_ENABLED=none`` 与 ``CHUNK_CONTEXT_HEADER=none``
    会得到**相反**的结果——同一个词，一个是"开"一个是"关"，且不报错。
    现已全部收敛到这里，由 ``tests/test_infra.py`` 的
    ``test_all_bool_configs_parse_identically`` 守着「所有 bool 配置对同一串输入的
    解析必须一致」。

    ``"none"`` 归入**假**：对布尔开关来说，"给了个空词"最危险的解读是"开"——
    一个手滑写成 ``=none`` 的配置会静默开启某个本不该开的功能；反向解读最多是
    功能没开、看得见。**取错误方向不对称的那一侧。**

    非空即真（``"1"/"yes"/"on"/"true"`` 及任何未列出的非空值 → True）。
    """
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "none")


def _env_list(key: str, default: str = "", sep: str = ",") -> list:
    """读取逗号分隔的列表型环境变量，自动去空白并丢弃空项。"""
    val = os.getenv(key)
    if val is None or val.strip() == "":
        raw = default
    else:
        raw = val
    return [item.strip() for item in raw.split(sep) if item.strip()]


# ============================================================
# 项目路径
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "logs"))
STATIC_DIR = Path(__file__).resolve().parent / "static"

# ============================================================
# 大模型配置（OpenAI 兼容协议：混元 / DeepSeek / 通义 / OpenAI 等通用）
# ============================================================
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(_env("LLM_TEMPERATURE", "0.1"))  # 低温度，保证回答严谨
LLM_TIMEOUT: int = int(_env("LLM_TIMEOUT", "60"))
LLM_MAX_TOKENS: int = int(_env("LLM_MAX_TOKENS", "1024"))
# 是否在流式请求中携带 stream_options（含 token 用量统计）。多数 OpenAI 兼容服务
# （如 Moonshot/Kimi）不支持该参数，会拒绝整个流式请求，故默认关闭以保证兼容；
# 需要 token 用量统计时打开即可。
LLM_STREAM_USAGE: bool = _env_bool("LLM_STREAM_USAGE", False)
# 是否关闭推理模型的「思考」阶段（默认关闭该开关，即保持模型默认行为）。
#
# 为什么需要它：本项目的默认模型 kimi-k2.6 是**推理模型**——它会在正文之前先产出
# 一段隐藏的 reasoning_content。实测（2026-09-11，同一问句）：
#     思考开启   首个可见字 11123ms | 推理 535 字 | 正文 1115 字 | finish=stop
#     思考关闭   首个可见字  1137ms | 推理   0 字 | 正文  875 字 | finish=stop
# 可见字延迟相差约 10 倍，且思考 token 与正文**共享 max_tokens 预算**，会把正文挤到
# 截断（实测多次出现 finish_reason=length）。企业知识库问答属「检索 + 简短归纳」，
# 不需要长链路推理，关掉思考是净收益。
#
# 为什么是配置项而不是代码里写死：该参数是**供应商特定**的（OpenAI 官方协议无此字段），
# 且并非所有模型都支持关闭——实测 kimi-k2.7-code 会返回 400
# 「invalid thinking: only type=enabled is allowed for this model」。换模型/换供应商时
# 必须能一键关掉，故由配置控制，代码不做任何模型名判断。
#
# 注意：关闭思考后该模型只接受 temperature=0.6（开启思考时只接受 1.0），
# 两者需配套修改，否则接口会以 400 明确拒绝——错误是自解释的。
LLM_DISABLE_THINKING: bool = _env_bool("LLM_DISABLE_THINKING", False)
# 自造的限流重试包装器（_RateLimitRetryModel）及其三个配置项
# （LLM_MAX_RETRIES / LLM_RETRY_BASE_DELAY / LLM_RETRY_MAX_DELAY）**已随
# 「改用不限流模型」一并移除**。模型现在直连 ChatOpenAI，重试由
# _build_raw_model 的 max_retries=0 明确关闭（快速失败、交给上层降级）。
# 理由见 app/providers/llm.py 的模块 docstring。
#
# 启动时对真实大模型做一次 ping 健康检查（默认关闭）。
# 关闭（默认）：跳过校验、直接判定为真实模式。理由是**可用性优先**——启动时网络
#   不通不该让服务起不来；配置格式本身仍会在启动时校验（不发请求、零成本），
#   而 Key / 模型名配错会在第一次提问时以自解释的报错立刻暴露。
# 开启：可提前发现 Key / 模型名错误并优雅降级为 Mock；代价是启动多 1~2 次调用与
#   1~2 秒。⚠️ 过去不敢开是因为"怕刷爆限频账号的配额"，**该顾虑已随账号更换消失**
#   ——若更看重"启动即发现配置错误"，现在打开它的代价比以前低得多。
LLM_HEALTH_CHECK: bool = _env_bool("LLM_HEALTH_CHECK", False)

# 是否启用真实大模型：未配置 Key 时自动降级为本地 Mock 模型（保证系统可离线跑通）
USE_REAL_LLM: bool = bool(LLM_API_KEY)

# ============================================================
# Agent（function calling）配置
#
# 决策权已交还模型：词面/语义打分、边际门控、灰区仲裁、档位选型均已删除。
# 被删机制的清单与理由见 app/core/agent.py 的模块 docstring。
# ============================================================
# 工具决策最多几轮 —— 单位是**模型调用轮数**，不是工具调用次数。
# 每一轮都是"模型看到上一轮的工具结果后重新决策"，可在这一轮内**并行发出多个
# 工具调用**（「查年假」+「查报销」各一条）。三种情形各需几轮：
#
#   ① 一轮并行收集                              1 轮
#   ② 链式：先查 A，才知道该拿哪个键去查 B        2 轮
#   ③ 参数被拒后自我纠正                         同轮内，或占用下一轮
#
# 故默认 2 已覆盖全部三种情形。
#
# 为什么从 3 降下来：第 3 轮实测**永远是同一件事——空转收口**。模型看到工具
# 结果后回一句「我查到了」（tool_calls=0），**而那一轮的文本会被 L4 覆盖**
# （最终答案必须由受控生成产出、带引用编号）。于是它唯一的产出就是"模型没有
# 更多工具调用"这个消息，代价却是一次完整的模型调用（实测 1.2~2.4s）。
# 统计 54 条真实工具链路：需要第 3 轮工具调用的是 **0 条**。
# 若将来出现三步链，把 TOOL_AGENT_MAX_STEPS 调回 3 即可。
#
# 注意：这项配置只被**工具 Agent**（app/core/tool_agent.py）消费，故叫
# TOOL_AGENT_MAX_STEPS。历史上它叫 AGENT_MAX_STEPS（单 Agent 时代），
# 五 Agent 架构下那个名字已无法说明"是哪个 Agent 的轮数"，故改名——旧名不再读取。
TOOL_AGENT_MAX_STEPS: int = int(_env("TOOL_AGENT_MAX_STEPS", "2"))

# ============================================================
# 复杂 RAG Agent（拆解 → 多次检索 → 合并）
# ============================================================
# 一次拆解最多产出几个子问题。子问题越多，检索次数与送入 L4 的片段越多，
# 成本与"上下文被稀释"的风险同步上升。提示词里要求 2~4 个，这里再兜一道上限
# （模型不遵守提示词是常态，兜底必须落在代码里而不是提示词里）。
COMPLEX_RAG_MAX_SUBQUERIES: int = int(_env("COMPLEX_RAG_MAX_SUBQUERIES", "4"))

# 多次检索合并去重后，最多送进 L4 几条片段。
# 为什么需要它：3 个子问题 × Top-K(5) 最坏可得 15 条，直接喂给 L4 会让
# 关键片段被大量弱相关片段稀释（引用编号也会膨胀到没人看）。默认 8 ≈
# 简单 RAG 的 1.6 倍，既体现"跨文档"的信息量，又不失控。
COMPLEX_RAG_MAX_DOCS: int = int(_env("COMPLEX_RAG_MAX_DOCS", "8"))


# ============================================================
# 混合意图路由（四层漏斗，见 app/core/routing/）
#
# 设计文档：docs/intent-routing-hybrid-design.md
#
# 这一组的默认值由三条前提决定，改任何一个默认值前请先读这三条：
#   ① 路由的首要目标是**减少 LLM 调用开销** —— 层①②③ 全程不调模型；
#   ② 不能引入过度的时间开销 —— 词面先判，判不了才升语义（惰性升级），
#      且每一步开始前都先看时间预算，超预算就**不启动**这一步；
#   ③ 识别不出时用 LLM 兜底 —— 层④ 是唯一的模型调用点，只在灰区发生。
#
# ⚠️ Phase 0 阶段只有预演接口（app/api/routing.py）读这些配置，
#    生产链路仍走 router_agent.route_query 的一次模型调用。
#    真正接管发生在 Phase 2（层①）/ Phase 3（层②③④），见文档 §8.1。
# ============================================================
# 是否允许从「词面」升级到「语义」。关掉后只跑词面 + 灰区 LLM 兜底，
# 省掉一次 embedding —— 用于回答"语义层到底值不值这笔延迟"。
ROUTE_SEMANTIC_ENABLED: bool = _env_bool("ROUTE_SEMANTIC_ENABLED", True)

# 灰区是否交给 LLM 仲裁（层④）。默认开：这正是"识别不出时的兜底"。
ROUTE_ARBITRATION_ENABLED: bool = _env_bool("ROUTE_ARBITRATION_ENABLED", True)

# 路由的**整体**时间预算（毫秒），是一个**硬上界**。
#
# 语义：路由从入口到出结论，最多花这么久。每一步动手之前先算
# `剩余 = 截止时刻 - 现在`，剩余不足就**不启动**这一步：
#   - 启动语义层前剩余不足 → 跳过，按词面继续；
#   - 启动仲裁前剩余不足 → 不调模型，直接保守兜底；
#   - 启动仲裁前剩余尚可   → 把仲裁的等待上限再收窄到 `min(自身上限, 剩余)`。
#
# 所以总耗时有证明：**≤ ROUTE_BUDGET_MS**。没有这条，最坏情况会变成
# "预算 + 各步各自的上限"之和，那个数字没人说得清（首版实测出现过 3.2s）。
#
# 默认 2500ms 的来由：今天生产链路每轮固定一次 chat 分类调用（实测 ~2s），
# 灰区路径 = 一次 embedding(~300ms) + 一次仲裁调用，与现状基本持平；
# 而占了绝大多数的锚定/词面路径是 **0~300ms**，比现状快一个数量级。
# 也就是说：预算买的是"最坏情况不比今天差"，省下来的是"最常见情况快得多"。
ROUTE_BUDGET_MS: int = int(_env("ROUTE_BUDGET_MS", "2500"))

# 单次语义升级（utterances 预热 + query embedding）的等待上限，是总预算的**子预算**。
# 默认 500ms：远程 embedding 通常 50~300ms，仍比一次 chat completion 便宜数倍。
#
# ⚠️ 首次冷启动要批量算全部 utterances 向量，可能超出这个上限。**超时不是浪费**：
#    被放弃的那个线程仍在跑，跑完会把缓存填上（进程内 + 文档向量持久化两级），
#    于是只有**第一个**请求降级，后续请求正常。这条比"启动时预热"更好——
#    启动自检不该被 embedding 服务拖住（项目为此吃过亏）。
ROUTE_EMBED_TIMEOUT_MS: int = int(_env("ROUTE_EMBED_TIMEOUT_MS", "500"))

# 层④ 灰区仲裁的等待上限，同样会被剩余预算进一步收窄。
# 给模型调用一个封顶，否则一次上游卡顿会把路由的延迟摊到 LLM_TIMEOUT（默认 60s）那么大。
ROUTE_ARBITRATION_TIMEOUT_MS: int = int(_env("ROUTE_ARBITRATION_TIMEOUT_MS", "2000"))

# 地板（层③ 的第一道）：**绝对证据**要够格才算"识别得出"。
#
# 口径：词面分 = query 与该能力**例句**的最大字符 n-gram Dice 相似度 ∈ [0, 1]
# （见 app/core/routing/similarity.py）。**注意量纲**：它以前是
# "Σ len(命中关键词)"（0~20+），2026-09-16 随词面打分一起换掉了。
# 沿用旧值 3.0 会让地板**永远通过**——那是彻底的静默失效（灰区判定形同虚设），
# 所以下面这个数是重新标定的，不是拍脑袋往下调。
#
# 为什么是 0.45（实测，29 条探针问句）：
#   判对项的最低分 0.375（"报销要走什么审批"）、判错项的最高分 0.400
#   （"年假是几天"被 leave_balance 抢走）。**两者重叠**，不存在能分开它们的
#   词面阈值——这正是"必须保留语义层"的量化证据。所以地板不取在重叠区里，
#   而是取在它**上方**：让这一整段模糊区一律**升级**到语义层，而不是由词面短路。
#   0.45 之上剩 14 条，全部判对（最低 0.533，对应"我想查下张三的工号"）。
#   设计取向：宁可多升一次语义，不可错得干脆（惰性升级的升级条件是门控说了算）。
ROUTE_LEXICAL_FLOOR: float = float(_env("ROUTE_LEXICAL_FLOOR", "0.45"))

# 边际（层③ 的第二道）：top 与**次优通道**的绝对证据差。
#
# 为什么边际不能用 RRF 融合分：RRF 分只由**排名**决定，rank1 与 rank2 的差
# 恒为 ~1/(k+1) ≈ 1.6%，与"领先一大截"还是"咬得很紧"完全无关。拿它比阈值，
# 等于要么永远进灰区、要么永远不进——两个都不是"边际"想表达的意思。
# 所以：**融合分只用来排序与展示，门控一律用同层同量纲的绝对差。**
#
# 0.25 的来历：词面能短路的那 14 条里，最小领先是 0.367（"王五的座机是多少"
# vs 语义上最接近的另一个通道）。取 0.25 给了它余量，同时凡是"领先不到
# 四分之一"的一律视为咬得很紧、交下一层。
ROUTE_LEXICAL_MARGIN: float = float(_env("ROUTE_LEXICAL_MARGIN", "0.25"))

# 语义层的地板与边际：显式配置优先，否则跟随实际 embedding 模式自适应。
# 两种向量的余弦尺度差一个数量级（本地哈希 ≈ 0.1~0.5，真实神经向量 ≈ 0.3~0.9），
# 套同一套阈值必然一边恒过、一边恒不过——与 effective_score_threshold 同一个坑。
_ROUTE_SEMANTIC_FLOOR_RAW = os.getenv("ROUTE_SEMANTIC_FLOOR", "")
ROUTE_SEMANTIC_FLOOR: Optional[float] = (
    float(_ROUTE_SEMANTIC_FLOOR_RAW) if _ROUTE_SEMANTIC_FLOOR_RAW.strip() else None
)
_ROUTE_SEMANTIC_MARGIN_RAW = os.getenv("ROUTE_SEMANTIC_MARGIN", "")
ROUTE_SEMANTIC_MARGIN: Optional[float] = (
    float(_ROUTE_SEMANTIC_MARGIN_RAW) if _ROUTE_SEMANTIC_MARGIN_RAW.strip() else None
)


def effective_route_semantic_floor() -> float:
    """语义地板的实际取值：显式配置优先，否则按 embedding 模式自适应。"""
    if ROUTE_SEMANTIC_FLOOR is not None:
        return ROUTE_SEMANTIC_FLOOR
    from app.utils.embedding import get_embedding_mode

    return 0.30 if get_embedding_mode() == "local-hash" else 0.35


def effective_route_semantic_margin() -> float:
    """语义边际的实际取值：显式配置优先，否则按 embedding 模式自适应。"""
    if ROUTE_SEMANTIC_MARGIN is not None:
        return ROUTE_SEMANTIC_MARGIN
    from app.utils.embedding import get_embedding_mode

    return 0.02 if get_embedding_mode() == "local-hash" else 0.03


# ============================================================
# 业务结构化数据库（SQLite，只读）
# ============================================================
# 存放员工 / 假期余额两张表，供 3 个只读工具查询。
# 初始化：`python scripts/seed_enterprise_db.py`
#
# 为什么是独立的 SQLite 而不是塞进向量库：这两类数据的**访问方式本质不同**——
# 制度文档是"语义相近就召回"，而「E1001 的年假余额」是**精确主键查询**。
# 用向量检索去查结构化事实，既慢又可能返回一个"看着像"的错误记录。
SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", str(DATA_DIR / "enterprise.db"))


# ============================================================
# Embedding 配置（优先复用大模型服务的 OpenAI 兼容 /embeddings 接口）
# ============================================================
EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "") or LLM_API_KEY
EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "") or LLM_BASE_URL
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
# 查询侧检索指令前缀（模型无关的配置项）：部分 embedding 模型（如 bge 系列）要求
# 查询拼接一段检索指令前缀，否则余弦相似度偏低、排序失真。是否拼接由本配置决定，
# 代码不识别任何具体模型名——换模型时清空该配置即可。
# 仅对 query 拼接，文档侧不加（标准 bge 用法）。
EMBEDDING_QUERY_PREFIX: str = os.getenv("EMBEDDING_QUERY_PREFIX", "")
# 降级方案：本地零依赖哈希向量维度（仅在无 Key 时使用）
LOCAL_EMBEDDING_DIM: int = int(_env("LOCAL_EMBEDDING_DIM", "1024"))

USE_REAL_EMBEDDING: bool = bool(EMBEDDING_API_KEY)

# ============================================================
# 向量数据库配置（milvus：分布式生产 / chroma：单机生产 / memory：零依赖内存库）
# ============================================================
#: 合法后端。改这里等于改「可选后端清单」，工厂与自检脚本都以此为准。
VECTOR_DB_CHOICES: tuple = ("memory", "chroma", "milvus")

# 读取时统一 strip + lower：.env 里写成 "Milvus" 或 " milvus " 也能识别。
# 不做归一化的话，大小写/空格写错会静静落进内存库分支，
# 表现得和「本来就配的内存库」一模一样，排查时没有任何线索。
VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "memory").strip().lower()

# 严格模式：开启后「配了 milvus/chroma 却连不上」会直接启动失败。
# 默认关闭，保留「降级而不是启动失败」的既有约定（前台可用性与后端配置问题
# 不互相掩盖）；但显式选了生产后端时，静默降级意味着数据被写进本地内存库、
# 而运维以为在用 Milvus —— 所以生产环境建议开启。
VECTOR_DB_STRICT: bool = _env_bool("VECTOR_DB_STRICT", False)

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "vector_store"))
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "enterprise_knowledge")

# ---- Milvus（仅 VECTOR_DB_TYPE=milvus 时生效）----
MILVUS_URI: str = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN: str = os.getenv("MILVUS_TOKEN", "")   # 本地免鉴权部署留空即可
MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "") or COLLECTION_NAME
# 向量维度：0 = 首次写入时按向量长度自动推断（推荐，换 embedding 模型不会踩维度不符）。
# 显式指定会与已有集合的维度做一致性校验，不一致时直接报错而不是静默写坏。
MILVUS_DIM: int = int(_env("MILVUS_DIM", "0"))
MILVUS_INDEX_TYPE: str = os.getenv("MILVUS_INDEX_TYPE", "HNSW")
MILVUS_METRIC_TYPE: str = os.getenv("MILVUS_METRIC_TYPE", "COSINE")


def effective_vector_collection() -> str:
    """当前**实际生效**的向量库集合名。

    各后端默认共用一个集合名，但 Milvus 允许用 ``MILVUS_COLLECTION`` 单独指定。
    配置快照（启动自检 / ``/health``）必须报生效的那个，否则运维看到的是
    「另一个后端的集合名」，排查「我到底连的是哪个集合」时会被带偏。
    """
    return MILVUS_COLLECTION if VECTOR_DB_TYPE == "milvus" else COLLECTION_NAME


def validate_vector_db_type() -> str:
    """校验并返回归一化后的向量库类型；非法值直接抛 ``ValueError``。

    ``VECTOR_DB_TYPE`` 写错（如 ``mlivus``）过去会静默落进内存库分支，
    于是「配置写错了」与「本来就配的内存库」表现完全一致，排查时毫无线索。
    这类**确定性的配置错误**应当立刻暴露，而不是被降级逻辑掩盖。
    """
    if VECTOR_DB_TYPE not in VECTOR_DB_CHOICES:
        raise ValueError(
            f"VECTOR_DB_TYPE={VECTOR_DB_TYPE!r} 不是合法后端；"
            f"可选值：{' | '.join(VECTOR_DB_CHOICES)}"
        )
    return VECTOR_DB_TYPE


# ============================================================
# Redis 配置（不可用时自动降级为进程内内存缓存）
# ============================================================
REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT: int = int(_env("REDIS_PORT", "6379"))
REDIS_DB: int = int(_env("REDIS_DB", "0"))
REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
SESSION_TTL: int = int(_env("SESSION_TTL", str(3600 * 24)))  # 会话 24h 过期

# ============================================================
# 对话与检索参数
# ============================================================
MAX_CHAT_HISTORY: int = int(_env("MAX_CHAT_HISTORY", "10"))  # 单会话最多保留 10 轮
SIMILARITY_TOP_K: int = int(_env("SIMILARITY_TOP_K", "5"))  # 单次检索返回 Top-K
# 余弦相似度阈值（0~1，越大越严格）。留空（None）时按"实际生效的 embedding 模式"自适应：
#   本地哈希向量 → 0.08（硬门禁，过滤字面不相关片段）；
#   真实神经向量（API）→ None，即「按相关度排序取 Top-K」而非硬门禁——
#   因为不同供应商分数尺度差异极大（text-embedding-3-small 常 0.3~0.9，而 bge-m3 常 0.05~0.15），
#   套用固定阈值会导致检索恒为空。排序取 Top-K 是唯一跨模型稳健的做法。
_SCORE_THRESHOLD_RAW = os.getenv("SCORE_THRESHOLD", "")
SCORE_THRESHOLD: Optional[float] = float(_SCORE_THRESHOLD_RAW) if _SCORE_THRESHOLD_RAW.strip() else None
# 软回退的最小相关度门槛：Top1 分数仍低于此值时，判定「知识库确实没有答案」，
# 返回空结果由生成节点给出诚实回答，而非硬凑无关片段（避免 RAG 幻觉）。
_FALLBACK_MIN_RAW = os.getenv("FALLBACK_MIN_SCORE", "")
FALLBACK_MIN_SCORE: Optional[float] = float(_FALLBACK_MIN_RAW) if _FALLBACK_MIN_RAW.strip() else None


def effective_score_threshold() -> Optional[float]:
    """检索阈值：显式配置优先，否则跟随实际 embedding 模式自适应。

    返回 None 表示「不按绝对分数过滤，直接按相关度排序取 Top-K」（API 模式默认）。
    """
    if SCORE_THRESHOLD is not None:
        return SCORE_THRESHOLD
    from app.utils.embedding import get_embedding_mode

    return 0.08 if get_embedding_mode() == "local-hash" else None


def effective_fallback_min() -> float:
    """软回退门槛：显式配置优先，否则跟随实际 embedding 模式自适应。"""
    if FALLBACK_MIN_SCORE is not None:
        return FALLBACK_MIN_SCORE
    from app.utils.embedding import get_embedding_mode

    return 0.05 if get_embedding_mode() == "local-hash" else 0.04

# ============================================================
# 融合权重与阈值（P1-1：此前是「伪配置」，已补齐为真实配置）
#
# 这一组曾经**只在消费方用 getattr(config, X, 默认值) 读取，而 config 里
# 根本没有定义** —— 结果改 .env 完全无效、只能改源码，与「配置驱动」的承诺
# 直接矛盾，且极易让人误以为「调过了没用」而得出错误结论。
# 现已补齐定义，并把消费方改为**运行时读取**，同时支持 .env 与运行期覆盖。
# ============================================================
RRF_K: int = int(_env("RRF_K", "60"))                       # RRF 平滑常数，越大排名差异越平缓
DENSE_WEIGHT: float = float(_env("DENSE_WEIGHT", "0.7"))    # 向量路权重
LEXICAL_WEIGHT: float = float(_env("LEXICAL_WEIGHT", "0.3"))  # 词面（BM25）路权重
# 查询改写：多一次 LLM 调用换取召回率，默认关（成本与延迟考量）
QUERY_REWRITE_ENABLED: bool = _env_bool("QUERY_REWRITE_ENABLED", False)
# 置信度低于此值且无工具结果时触发优雅拒答（宁可说不知道，也不要编）。
#
# 它是「证据够不够」的**唯一**基准线。此前这条线被三处共用：拒答、级联升档下限、
# 动态路由升级下限——级联与路由已随「改用不限流模型 + function calling」删除，
# 现在只剩拒答一个消费方，标度也跟着回归到一个普通的绝对分数。
REFUSE_THRESHOLD: float = float(_env("REFUSE_THRESHOLD", "0.25"))
# L1 预处理：短于此长度的文档视为噪声丢弃；近重复 Jaccard 阈值
PREPARE_MIN_DOC_CHARS: int = int(_env("PREPARE_MIN_DOC_CHARS", "10"))
PREPARE_NEAR_DUP_THRESHOLD: float = float(_env("PREPARE_NEAR_DUP_THRESHOLD", "0.95"))
# 用户反馈落盘位置（L5 评估消费）
FEEDBACK_FILE: Path = Path(os.getenv("FEEDBACK_FILE", "") or (LOG_DIR / "feedback.jsonl"))

# ============================================================
# 词面倒排索引配置（BM25，见 app/rag/lexical.py）
# ============================================================
# BM25 参数：k1 控制词频饱和，b 控制文档长度归一化强度
LEXICAL_BM25_K1: float = float(_env("LEXICAL_BM25_K1", "1.5"))
LEXICAL_BM25_B: float = float(_env("LEXICAL_BM25_B", "0.75"))

# ============================================================
# Rerank 精排（可选增强，见 app/rag/rerank.py）
# ============================================================
# cross-encoder 对召回候选做二次评分重排，真正提升「该进 Top-K 却排在后面」的片段。
# 与 reorder（零模型交错重排）正交：rerank 用模型打分改善排序，代价是一次前向（CPU 约 80~120ms）。
# 默认关闭：需额外依赖 sentence-transformers，未安装时自动降级为「不精排」，绝不拖垮检索。
RERANK_ENABLED: bool = _env_bool("RERANK_ENABLED", False)
# 中文场景常用 BGE 系列重排模型（也支持任意 sentence-transformers 兼容 cross-encoder）
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
# 参与精排的候选条数：先粗召回 Top-N 交给 cross-encoder，精排后再截 Top-K
RERANK_TOP_N: int = int(_env("RERANK_TOP_N", "10"))

# ============================================================
# 可观测性接入（LangSmith，可选，见 app/core/observability.py）
# ============================================================
# 设置 LANGCHAIN_TRACING_V2 / API Key / Project 后，LangChain 的每次 LLM 调用
# 自动上报 trace 到 LangSmith 控制台，业务代码零改动。需 pip install langsmith。
# 关键约束：必须在「第一次 LLM 调用」前设置，故在 llm_factory 导入时执行（幂等）。
LANGSMITH_ENABLED: bool = _env_bool("LANGSMITH_ENABLED", False)
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "langgraph-enterprise-bot")
# 自托管 LangSmith 时填写；留空走官方 SaaS（smith.langchain.com）
LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT", "")

# ============================================================
# Embedding 缓存配置（见 app/utils/cache.py）
# ============================================================
EMBEDDING_CACHE_QUERY_TTL: float = float(_env("EMBEDDING_CACHE_QUERY_TTL", "600"))  # 查询向量 TTL（秒）
EMBEDDING_CACHE_QUERY_MAX_SIZE: int = int(_env("EMBEDDING_CACHE_QUERY_MAX_SIZE", "1000"))  # 查询缓存容量

# ============================================================
# 入站限流配置（见 app/core/rate_limit.py）
# ============================================================
RATE_LIMIT_PER_MINUTE: int = int(_env("RATE_LIMIT_PER_MINUTE", "20"))  # 单 IP 每分钟请求上限

# ============================================================
# 记忆系统配置（短期记忆 + 长期记忆）
# ============================================================
# 长期记忆落盘目录（SOUL.md / USER.md / MEMORY.md / history.jsonl）
MEMORY_DIR: Path = Path(os.getenv("MEMORY_DIR", BASE_DIR / "memory_store"))
MEMORY_ENABLED: bool = _env_bool("MEMORY_ENABLED", True)

# 短期记忆：注入 prompt 的最近对话轮数（超出部分走压缩归档）
SHORT_TERM_WINDOW: int = int(_env("SHORT_TERM_WINDOW", "6"))
# 短期记忆字符预算：防止长对话把上下文撑爆
SHORT_TERM_MAX_CHARS: int = int(_env("SHORT_TERM_MAX_CHARS", "2000"))

# 整理器（Consolidator）：把「尚未归档的新增消息」压缩后追加到 history.jsonl。
# 判据是**新增条数**，不是历史总长度——历史列表被 ltrim 封顶，会话饱和后
# 长度恒为常数，用总长度做阈值会退化成「每一轮都触发」（每轮一次模型调用）。
# CONSOLIDATE_THRESHOLD 比的是**单次真正压缩的条数**，直接对应模型调用成本：
# 默认 10 条 + 尾部保留 4 条 ⇒ 每约 12 条新消息（6 轮）压缩一次。
CONSOLIDATE_ENABLED: bool = _env_bool("CONSOLIDATE_ENABLED", True)
CONSOLIDATE_THRESHOLD: int = int(_env("CONSOLIDATE_THRESHOLD", "10"))  # 单次归档至少要压缩的消息条数
CONSOLIDATE_KEEP_TAIL: int = int(_env("CONSOLIDATE_KEEP_TAIL", "4"))  # 尾部保留不归档的条数（约 2 轮）

# 蒸馏（Dream）：把 history.jsonl 的归档沉淀进长期记忆文件。
# 归档累积到 DREAM_BATCH_SIZE 条时自动执行一次，也支持 API 手动触发。
# 成本受两道闸门控制：① 归档本身要攒够 CONSOLIDATE_THRESHOLD 条才产生；
# ② 蒸馏要再攒够一整批。因此一次蒸馏摊薄到很多轮对话上。
DREAM_ENABLED: bool = _env_bool("DREAM_ENABLED", True)
DREAM_BATCH_SIZE: int = int(_env("DREAM_BATCH_SIZE", "5"))  # 单次蒸馏消费的归档条目数，兼作自动触发阈值
MAX_HISTORY_ENTRIES: int = int(_env("MAX_HISTORY_ENTRIES", "1000"))  # history.jsonl 保留上限

# 长期记忆注入 prompt 时的字符上限（防止记忆膨胀挤占检索上下文）
LONG_TERM_MAX_CHARS: int = int(_env("LONG_TERM_MAX_CHARS", "1500"))

# ============================================================
# 切片策略：300 字符分片 + 60 字符重叠
# ============================================================
CHUNK_SIZE: int = int(_env("CHUNK_SIZE", "300"))
CHUNK_OVERLAP: int = int(_env("CHUNK_OVERLAP", "60"))
SEPARATORS: list = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

# ============================================================
# 切片策略（切分改造 T1-1）：配置化 + 策略可替换
#
# 设计原则（见切分改造任务清单 P5）：
#   **回退 = 改配置，不改代码**。以下所有新增项的默认值都刻意保持
#   「与改造前完全一致」的行为，新增逻辑默认关闭；未启用时
#   `tests/test_chunking_baseline.py` 必须仍然通过。
# ============================================================
# 切分策略：recursive | structure（结构感知，T3-2 实现）
#
# 默认值为 structure（T5-2：移除 recursive 的「默认特殊地位」）。
# 此前默认值是 recursive，导致一个隐蔽陷阱：**漏配 .env 时会静默回落到旧策略**，
# 与线上实际行为不一致，且没有任何报错。现在代码默认与线上保持一致。
#
# recursive 的实现**永不删除**：它是无结构（flat）文档的降级路径，
# 这是设计约束而非临时兼容 —— 见 docs/chunking-contract.md。
CHUNK_STRATEGY: str = os.getenv("CHUNK_STRATEGY", "structure").strip().lower()

# 结构感知策略的三参数（与 Haystack DocumentSplitter 的
# split_length / split_threshold / split_overlap 同构）
CHUNK_TARGET_CHARS: int = int(_env("CHUNK_TARGET_CHARS", "220"))      # 目标块长（贪心累加阈值）
CHUNK_HARD_MAX_CHARS: int = int(_env("CHUNK_HARD_MAX_CHARS", "340"))  # 硬上限，超出才按句二次切
CHUNK_MIN_CHARS: int = int(_env("CHUNK_MIN_CHARS", "60"))             # 下限，低于此合并到同章相邻节

# 上下文头：块正文前置 `{doc_title} > {chapter} > {section}`（T3-3）
# 注意：头不参与 chunk_chars 计量，避免污染块长统计与阈值判断
CHUNK_CONTEXT_HEADER: bool = _env_bool("CHUNK_CONTEXT_HEADER", False)

# 结构元数据增强：heading_path / chapter / section 等（T3-4）
CHUNK_ENRICH_METADATA: bool = _env_bool("CHUNK_ENRICH_METADATA", False)

# 降级（structure=flat 文档）走递归路径时的 overlap 比例。
# 生效点：app/rag/indexer.py::_fallback_overlap —— 降级路径的实际重叠 =
# round(CHUNK_SIZE * 本值)，常规递归路径仍用 CHUNK_OVERLAP，两者互不影响。
# 想精确复现「接线前」行为（降级路径也用 CHUNK_OVERLAP），把本值设为
# CHUNK_OVERLAP / CHUNK_SIZE 即可（当前默认配置下即 0.2）——回退靠改配置，不改代码。
CHUNK_FALLBACK_OVERLAP_RATIO: float = float(_env("CHUNK_FALLBACK_OVERLAP_RATIO", "0.12"))

# 短片段治理阈值（原为 getattr 兜底 40，现显式配置化，取值保持 40 不变）
INDEX_MIN_CHUNK_CHARS: int = int(_env("INDEX_MIN_CHUNK_CHARS", "40"))

# ------------------------------------------------------------
# 结构解析锚点（T2-1 消费）｜模型无关，纯正则，全部可配置
# 多个模式用 "|" 分隔
# ------------------------------------------------------------
# L1 章：第三章 假期管理 / 一、账号与密码
STRUCTURE_CHAPTER_PATTERNS: str = os.getenv(
    "STRUCTURE_CHAPTER_PATTERNS",
    r"第[一二三四五六七八九十百]+章|^[一二三四五六七八九十]+、",
)
# L2 节：3.2 年休假 / Q8：...
STRUCTURE_SECTION_PATTERNS: str = os.getenv(
    "STRUCTURE_SECTION_PATTERNS",
    r"^\d+\.\d+|^Q\d+[：:]",
)
# L3 条：1、第一步 / （1）
STRUCTURE_ITEM_PATTERNS: str = os.getenv(
    "STRUCTURE_ITEM_PATTERNS",
    r"^\d+[、.]|^[（(]\d+[）)]",
)
# Markdown 标题：## 6 知识库管理 API
STRUCTURE_MD_PATTERNS: str = os.getenv("STRUCTURE_MD_PATTERNS", r"^#{1,6}\s")

# 无结构文档（L1+L2 锚点命中数为 0）是否降级走递归路径
STRUCTURE_FLAT_FALLBACK: bool = _env_bool("STRUCTURE_FLAT_FALLBACK", True)

# 按文档灰度（T5-1）：仅列出的文件名走 CHUNK_STRATEGY，其余走 recursive
# 逗号分隔，留空表示全部生效
CHUNK_STRATEGY_SCOPE: list = _env_list("CHUNK_STRATEGY_SCOPE", "")

# 父子双层索引开关（P1，T6-1；当前语料收益有限，默认关）
PARENT_CHUNK_ENABLED: bool = _env_bool("PARENT_CHUNK_ENABLED", False)

# 父块合并上限：同章内连续子块累计到此长度即切一个父块。
# 上限越大上下文越完整，但注入 prompt 的 token 越多；1000 字≈2~3 个子块。
PARENT_MAX_CHARS: int = int(_env("PARENT_MAX_CHARS", "1000"))

# 父块旁路存储路径（父块不进向量库，只按 parent_id 回捞，故用 JSON 即可）
PARENT_STORE_PATH: str = os.getenv("PARENT_STORE_PATH", "vector_store/parents.json")

# ============================================================
# PDF 图片抽取（T6-2，默认关）
# ============================================================
# 默认关闭的理由：多数企业 PDF 的图片是**装饰性**的（logo / 图标 / 分隔线 / 水印），
# 把它们的占位描述塞进文本反而污染切分与检索。只有「图片内含关键信息」
# 的 PDF（架构图、数据图表、扫描件）才值得开启。
PDF_IMAGE_EXTRACTION: bool = _env_bool("PDF_IMAGE_EXTRACTION", False)

# 面积下限（PDF pt²，1pt≈1/72 英寸）：小于此值的图片视为装饰元素直接丢弃。
# 10000 pt² ≈ 100×100 pt。这是避免「图标污染文本」的关键闸门。
PDF_IMAGE_MIN_AREA: int = int(_env("PDF_IMAGE_MIN_AREA", "10000"))

# 是否对抽取的图片做 OCR（需要 pytesseract + 系统 tesseract，缺一即优雅降级跳过）
PDF_IMAGE_OCR: bool = _env_bool("PDF_IMAGE_OCR", False)

# 图片落盘目录（按文档名分子目录）
PDF_IMAGE_DIR: str = os.getenv("PDF_IMAGE_DIR", "artifacts/pdf_images")

# 渲染分辨率：越高 OCR 越准，但文件越大、耗时越久
PDF_IMAGE_DPI: int = int(_env("PDF_IMAGE_DPI", "150"))

# ============================================================
# Dify 兼容接口（外部知识库 · T-DIFY）
# ============================================================
# 让 Dify 把本项目当作「外部知识库」调用：Dify 的知识检索节点会
# POST {base_url}/retrieval 并带上 Bearer 鉴权。
# 规范见 https://docs.dify.ai/zh/self-host/use-dify/knowledge/external-knowledge-api
#
# 鉴权密钥由**本项目**校验（Dify 只负责透传）。留空 = 端点拒绝服务（安全默认），
# 避免未鉴权就把整个知识库暴露出去。
DIFY_API_KEY: str = os.getenv("DIFY_API_KEY", "").strip()

# 期望的 knowledge_id。留空 = 接受任意非空值（单知识库场景，最省事）；
# 填写后，Dify 传来的 knowledge_id 必须相等，否则返回 2001。
DIFY_KNOWLEDGE_ID: str = os.getenv("DIFY_KNOWLEDGE_ID", "").strip()

# 上传文件大小上限（字节）。multipart 上传会先预检 Content-Length，
# 再分块读取并在超限时立即中断 —— 不能先全量读进内存再判断。
MAX_UPLOAD_BYTES: int = int(_env("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 默认 10MB

# 单篇文档参与索引的最大字符数。**数字只在这里定义一次**——
# `app/utils/validator.py` 的 `MAX_DOC_CONTENT` 由它派生，不再各写一个 50_000。
# 这是「防呆」上限：切分后块数随字符数线性增长，无限灌入会拖垮检索与成本。
#
# ⚠️ 两个上传入口对**超限的处理不同，而且这个不同是有意的**：
#   - `POST /knowledge/upload`（JSON，给前端与第三方系统调）：**拒收**并说明超了多少。
#     调用方是程序，能据此修正后重试；
#   - `POST /knowledge/upload/file`（multipart，给人拖文件用）：**截断**，并在响应体里
#     回 `truncated: true`、日志记明截了多少。
#     人对着一份 300 页 PDF，直接报错等于整篇都不收；截断 + 显式告知才是有用的行为。
# 共同点是**上限这个数字只有一个来源**；差异只是"超限之后怎么办"，
# 且两处都必须把结果**显式告知**调用方（拒收有 error，截断有 truncated 字段）。
MAX_DOC_CONTENT_CHARS: int = int(_env("MAX_DOC_CONTENT_CHARS", "50000"))

# CORS 允许的来源（逗号分隔）。默认通配以方便本地调试；
# 部署到具体域名时应显式列出（如 https://kb.example.com），
# 此时会自动允许携带凭据；来源为通配时凭据自动关闭（浏览器规范要求）。
CORS_ORIGINS: list = _env_list("CORS_ORIGINS", "*")

# ============================================================
# 入站鉴权（P0-3）
#
# 默认关闭以保证「零行为变化」；**部署到任何非本机环境前必须开启**。
# 开启后：除免鉴权路径外，所有请求须带 X-API-Key 或 Authorization: Bearer <key>。
#
# 安全默认（fail-closed）：开了 AUTH_ENABLED 却没配 AUTH_API_KEY 时，
# 一律拒绝而非放行 —— 否则「开着鉴权但谁都能过」比不开更危险，
# 因为管理员会误以为已经防护过了。
# ============================================================
AUTH_ENABLED: bool = _env_bool("AUTH_ENABLED", False)
AUTH_API_KEY: str = os.getenv("AUTH_API_KEY", "").strip()
# 免鉴权路径：运维探针 + 自带鉴权的端点。
#   /health、/            存活探针（容器内 HEALTHCHECK 依赖）
#   /docs、/openapi.json  接口文档（不含数据，按需从列表移除以收紧）
#   /dify/info            接入自检探针，只返回配置状态不含密钥
#   /retrieval、/dify/retrieval  由 Dify 模块用自己的 DIFY_API_KEY 校验
AUTH_EXEMPT_PATHS: list = _env_list(
    "AUTH_EXEMPT_PATHS",
    "/health,/,/docs,/redoc,/openapi.json,/dify/info,/retrieval,/dify/retrieval",
)

# ============================================================
# 来源访问控制（按用户隔离知识库，承接 retrieve() 的 allowed_sources 参数）
#
# 把「用户身份」解析为「允许访问的文档来源白名单」，使权限参数真正端到端生效。
# 配置方式（.env，JSON）：SOURCE_ACL={"alice":["hr/*"],"bob":["finance/*"],"*":["*"]}
#   - 键为 user_id；特殊键 "*" 作为「默认规则」（未匹配到具体用户时适用）。
#   - 值为来源模式列表，支持 fnmatch 通配（如 "hr/*" 匹配 hr/ 开头的全部来源）。
#   - 值中含 "*" → 表示「允许访问全部来源」（不限制）。
#   - ACL 留空 {} → 不做来源限制（与无 ACL 部署行为一致）。
#   - 配了 ACL 却既无该用户、也无 "*" 默认规则 → fail-closed 全部拒绝（返回 []）。
# 服务端解析结果对客户端传入的 allowed_sources 具有**权威性**：ACL 生效时以
# 服务端为准，客户端无法放宽权限（最多只能由服务端进一步收窄，见 chat.py）。
# ============================================================
def _parse_source_acl(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        data = __import__("json").loads(raw)
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict = {}
    for k, v in data.items():
        out[str(k)] = [str(x) for x in v] if isinstance(v, (list, tuple)) else []
    return out


SOURCE_ACL: dict = _parse_source_acl(os.getenv("SOURCE_ACL", ""))

# ============================================================
# 服务配置
# ============================================================
APP_PORT: int = int(_env("APP_PORT", "8001"))
APP_TITLE: str = "企业智能助手 · RAG 知识引擎"
APP_VERSION: str = "1.0.0"


def mask_secret(secret: str) -> str:
    """脱敏展示密钥，避免日志泄露。"""
    if not secret:
        return "<未配置>"
    if len(secret) <= 8:
        return secret[:2] + "*" * (len(secret) - 2)
    return f"{secret[:6]}{'*' * 8}{secret[-4:]}"


def dump_config() -> dict:
    """输出可安全打印的配置快照（用于启动自检与 /health 接口）。"""
    return {
        "llm": {
            "mode": "real" if USE_REAL_LLM else "mock",
            "base_url": LLM_BASE_URL,
            "model": LLM_MODEL_NAME,
            "temperature": LLM_TEMPERATURE,
            "timeout": LLM_TIMEOUT,
            "max_tokens": LLM_MAX_TOKENS,
            # 自造重试已移除；这里报的是 _build_raw_model 实际生效的值
            "max_retries": 0,
            "disable_thinking": LLM_DISABLE_THINKING,
            "api_key": mask_secret(LLM_API_KEY),
        },
        "embedding": {
            "mode": "real" if USE_REAL_EMBEDDING else "local-hash",
            "base_url": EMBEDDING_BASE_URL,
            "model": EMBEDDING_MODEL_NAME if USE_REAL_EMBEDDING else f"local-hash-{LOCAL_EMBEDDING_DIM}d",
        },
        "vector_db": {"type": VECTOR_DB_TYPE, "collection": effective_vector_collection()},
        "redis": {"host": REDIS_HOST, "port": REDIS_PORT, "db": REDIS_DB},
        "rag": {
            "top_k": SIMILARITY_TOP_K,
            "score_threshold": effective_score_threshold(),
            "fallback_min_score": effective_fallback_min(),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "rerank_enabled": RERANK_ENABLED,
            "rerank_model": RERANK_MODEL if RERANK_ENABLED else None,
        },
        "observability": {
            "langsmith_enabled": bool(LANGSMITH_ENABLED and LANGSMITH_API_KEY),
            "langsmith_project": LANGSMITH_PROJECT,
        },
        "chat": {"max_history": MAX_CHAT_HISTORY, "session_ttl": SESSION_TTL},
        "agents": {
            # 五个 Agent 的可调项集中在这里。工具清单的唯一来源是
            # app/core/tool_agent.py::AGENT_TOOLS，不在这里再列一份——
            # 两处清单必然漂移，且是静默的。
            "router": {"enabled": True, "fallback": "simple_rag"},
            "smalltalk": {"mode": "template"},
            "complex_rag": {
                "max_subqueries": COMPLEX_RAG_MAX_SUBQUERIES,
                "max_docs": COMPLEX_RAG_MAX_DOCS,
            },
            "tool": {"max_steps": TOOL_AGENT_MAX_STEPS},
            "refuse_threshold": REFUSE_THRESHOLD,
        },
    }
