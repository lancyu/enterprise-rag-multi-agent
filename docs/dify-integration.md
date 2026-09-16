# Dify 接入指南（外部知识库）

本项目实现 Dify 的 **External Knowledge API** 契约，可被 Dify 的
「知识检索」节点直接调用。

```
Dify 应用（工作流 / 对话流）
    │  知识检索节点
    │  POST {base_url}/retrieval
    │  Authorization: Bearer <DIFY_API_KEY>
    ▼
本项目（FastAPI · 端口 8001）
    ├─ L3 检索：向量 + 词面 + 改写三路召回 → RRF 融合
    ├─ 切分策略：structure（章节感知）
    └─ 返回 {"records": [{content, score, title, metadata}]}
```

**为什么要这么做**：文档不用再往 Dify 传一份。切分策略、检索权重、
拒答阈值、父子索引全部仍由本项目掌控，Dify 只负责编排与生成。
数据不搬家，策略不分散。

---

## 1. 服务端配置（本项目）

在 `.env` 里加两项：

```bash
# 鉴权密钥：Dify 会原样透传，**由本项目校验**。留空 = 端点拒绝服务（安全默认）。
DIFY_API_KEY=换成一个足够长的随机串

# 期望的 knowledge_id。留空 = 接受任意非空值（单知识库场景最省事）。
DIFY_KNOWLEDGE_ID=
```

生成密钥示例：

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

改完重启服务，然后自检：

```bash
curl --noproxy '*' http://localhost:8001/dify/info
```

关注两个字段：
- `enabled: true` —— 说明 `DIFY_API_KEY` 已生效
- `index_chunks > 0` —— 说明索引已建立；**为 0 时 Dify 永远检索不到内容**

---

## 2. Dify 侧配置

在 Dify 工作台：**知识库 → 外部知识库 → 连接外部知识库**，填写：

| 字段 | 填什么 | 说明 |
|---|---|---|
| Name | 例如「企业知识引擎」 | 显示名，随意 |
| API Base URL | `http://<本项目地址>:8001` 或 `http://<本项目地址>:8001/dify` | 两种都行，见下 |
| API Key | 与 `DIFY_API_KEY` 一致 | Dify 会放进 `Authorization: Bearer` |
| 外部知识库 ID | 任意字符串（如 `enterprise-kb`） | 会作为 `knowledge_id` 传来；服务端配了 `DIFY_KNOWLEDGE_ID` 则必须一致 |

> **URL 拼法**：Dify 会在你填的地址后**自动追加 `/retrieval`**。
> 本项目同时注册了 `/retrieval` 和 `/dify/retrieval`，所以
> 填 `http://host:8001` 或 `http://host:8001/dify` 都能通。

⚠️ **网络可达性**：Dify 必须能访问到本项目。
- 若 Dify 跑在 Docker 里，用 `host.docker.internal:8001`（Mac/Windows）或宿主机内网 IP；
- 若 Dify 在另一台机器 / 云端，需要内网穿透或把服务暴露到可达地址。
这是「配置都对但连不上」最常见的原因，先 `curl` 通再配 Dify。

---

## 3. 接口契约

请求：

```bash
curl --noproxy '*' -X POST http://localhost:8001/retrieval \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <DIFY_API_KEY>" \
  -d '{
    "knowledge_id": "enterprise-kb",
    "query": "年假有多少天",
    "retrieval_setting": {"top_k": 3, "score_threshold": 0.0}
  }'
```

响应：

```json
{
  "records": [
    {
      "content": "【章节】第三章 假期管理 > 3.2 年休假 ...",
      "score": 0.9905,
      "title": "员工手册.txt",
      "metadata": {
        "source": "data/员工手册.txt",
        "file_name": "员工手册.txt",
        "chapter": "第三章 假期管理",
        "section": "3.2 年休假",
        "heading_path": "第三章 假期管理 > 3.2 年休假",
        "chunk_index": 3
      }
    }
  ]
}
```

错误响应（HTTP 200 + 错误码，便于 Dify 界面直接展示原因）：

| error_code | 含义 | 排查 |
|---|---|---|
| `1001` | Authorization 头缺失或格式不对 | 必须是 `Bearer <key>` |
| `1002` | 密钥错误 / 服务端未配置 `DIFY_API_KEY` | 看服务端日志 |
| `2001` | knowledge_id 为空或不匹配 | 对照 `DIFY_KNOWLEDGE_ID` |
| `5001` | 检索内部异常（HTTP 500） | 看服务端日志 |

---

## 4. 三个已处理的坑（改代码时别踩回去）

### 4.1 score 必须归一化到 0~1 ⚠️ 最容易踩

Dify 的 `score_threshold` 是 **0~1 的绝对相关度**语义。
而本项目内部用的是 RRF 融合分，量级只有
`(w_dense + w_lex) / (RRF_K + 1)` ≈ **0.016**。

若直接透传，Dify 里只要把分数阈值调到 0.02 以上，
**所有结果都会被过滤掉** —— 表现为「知识库明明有内容，Dify 却永远检索不到」，
且很难从 Dify 侧看出原因。

适配层已除以 RRF 理论上限做归一化（见 `app/api/dify.py::_normalize_score`），
与项目内置信度计算口径一致，跨查询可比。

### 4.2 metadata 不能是 null

Dify 文档明确：`metadata` 必须是对象，**为 null 会导致 Dify 检索流程报错**。
适配层已保证即使无可输出字段也返回 `{}`。

### 4.3 入站限流

`/retrieval` 走全局 IP 限流中间件（与业务接口一致）。
若 Dify 侧调用密集被限流（HTTP 429），在 `app/main.py` 的限流白名单里
按需放行，或调高限流阈值。`/dify/info` 作为运维探针已免限流。

---

## 5. 排错清单

| 现象 | 先看 |
|---|---|
| Dify 报连接失败 | 从 Dify 所在机器 `curl` 本项目 `/dify/info` 是否可达 |
| 报 `1002` | 服务端 `.env` 是否配了 `DIFY_API_KEY` 并重启 |
| 能连上但永远 0 条 | `/dify/info` 的 `index_chunks` 是否 > 0（为 0 先 `build_index()`） |
| 有结果但太少 | 把 Dify 的 Score Threshold 设为 0（本项目已归一化，阈值 0.5 会滤掉大半） |
| 结果不相关 | 用本项目的 `/knowledge/search` 对比；若本项目侧也差，是检索问题不是适配问题 |

---

## 6. 当前边界

- 只提供**检索**能力（外部知识库契约）。对话/生成仍由 Dify 负责。
- 除了「知识检索节点」，还可以把本项目当成 Dify **自定义工具**（让 Agent 自主决定
  何时检索）：在 Dify 的自定义工具里导入 `http://<host>:8001/dify/openapi.json` 即可，
  鉴权同样是 Bearer。两条路指向同一个 `/retrieval`。
- 想让 Dify 调用本项目的**完整问答链路**（含生成与拒答）目前不支持 ——
  该能力会与 Dify 自身的生成节点职责重叠，有明确需要时再补。
- 父子双层索引若开启，返回给 Dify 的 `content` 会自动用父块（上下文更完整），
  开关仍由本项目的 `PARENT_CHUNK_ENABLED` 控制。
