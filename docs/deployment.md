# 部署与运维手册

> 面向接手人：从零把服务跑起来，并知道它**不能**怎么跑。
>
> 适用定位：**单实例、内网、有专人维护**。这个前提贯穿全文——
> 它是本服务能稳定运行的边界，不是一句客套话，理由见第一节。

配套文档：功能与接口见 `README.md`；架构与行号级说明见 `docs/project-introduction.md`；
Dify 接入见 `docs/dify-integration.md`。

---

## 一、先读这一节：单实例限制

**本服务不是无状态服务。** 有五处状态是模块级单例或进程内字典：

| 状态 | 所在位置 | 多副本下的后果 |
|---|---|---|
| 会话历史 | Redis 不可用时降级为 `MemoryRedis`（`app/db/redis_db.py`） | 同一会话被分到不同副本 → 历史时有时无，多轮直接断 |
| 内存向量库 | `VECTOR_DB_TYPE=memory` | 每个副本一份索引 → 同一问题返回不同结果 |
| 词面倒排索引 | BM25（`app/rag/lexical.py`） | 同上 |
| Embedding 缓存 | `app/utils/cache.py`（查询 LRU + 文档哈希） | 命中率归零 → Embedding 调用量被放大 |
| 入站限流计数 | `app/core/rate_limit.py`（进程内滑动窗口） | 每个副本各算一份 → 实际放行量 ×N，限流形同虚设 |

> 动态路由删除后（见 `docs/multi-agent-architecture.md`），
> 原先的「会话 pin 表」「级联预算」「路由统计」三处状态已不存在；
> 上面这张表是**当前**的完整清单。

这五处的共同特征是**不会报错**：多副本部署后服务照常返回 200，
只是会话时有时无、限流失效、上游配额被偷偷用光。
**没有任何报错，只有不正确的行为**——这是它比崩溃更难排查的原因。

### 结论

| 做法 | 是否可行 |
|---|---|
| 单进程 + 多 CPU 核（uvicorn 默认单 worker） | ✅ 推荐 |
| `uvicorn --workers N` | ❌ 会触发上述全部问题 |
| compose 里 `deploy.replicas > 1` | ❌ 同上 |
| Kubernetes 多副本 | ❌ 除非先外置状态 |

### 为什么"粘性会话"不能作为绕过方案

粘性会话（session affinity）只能解决会话历史一处。
限流计数与 Embedding 缓存仍是**每副本各自维护**，副本数越多偏差越大。
要让多副本真正成立，必须把上述状态外置到 Redis——
项目已经把 Redis 用在缓存上，但没有用在会话状态上。

> 单实例不是能力不足，而是**当前架构下唯一正确的部署形态**。
> 要突破它，请先做状态外置，而不是先加副本。

---

## 二、环境准备

### 2.1 依赖

| 组件 | 版本 | 必需性 |
|---|---|---|
| Python | 3.11+ | 必需 |
| Redis | 7.x | 建议（缓存/任务票据；不可用时有降级路径） |
| Milvus | v2.4.x（Standalone） | 可选，仅 `VECTOR_DB_TYPE=milvus` |
| etcd + MinIO | 随 Milvus | 选 Milvus 时必需（Milvus 自身不存元数据与对象） |

```bash
# 本地直跑
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-vector.txt   # 仅用 chroma / milvus 时需要
```

### 2.2 配置 `.env`

```bash
cp .env.example .env
```

**必填项**（不填则退化为本地 Mock 模型，问答质量不具备参考价值）：

| 变量 | 说明 |
|---|---|
| `LLM_API_KEY` | 大模型 API Key |
| `LLM_BASE_URL` | OpenAI 兼容端点 |
| `LLM_MODEL_NAME` | 模型名 |

**公网部署必填**（内网可缓，但要知道它们的存在）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `AUTH_ENABLED` | `false` | 打开后所有请求需带 API Key |
| `AUTH_API_KEY` | 空 | **fail-closed**：开了鉴权却留空 → 拒绝所有请求 |
| `AUTH_EXEMPT_PATHS` | 见 `.env.example` | 免鉴权路径白名单 |
| `CORS_ORIGINS` | `*` | 生产环境务必改成具体域名 |
| `RATE_LIMIT_PER_MINUTE` | `20` | 单 IP 每分钟上限。**填 `0` 是"全拒"不是"不限"** |
| `SOURCE_ACL` | 空 | 按来源/部门做知识隔离 |

**向量库相关**（最容易踩坑的一组）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `VECTOR_DB_TYPE` | `memory` | `memory` / `chroma` / `milvus` |
| `VECTOR_DB_STRICT` | `false` | **建议生产设为 `true`**：配了 milvus 却连不上时直接拒绝启动，而不是静默降级为内存库 |
| `MILVUS_URI` | `http://localhost:19530` | 容器内用服务名 `milvus` |

> ⚠️ `VECTOR_DB_TYPE=milvus` 但镜像里没装客户端包（构建时未传
> `WITH_VECTOR_CLIENTS=1`）时，服务会**照常启动**并把数据写进内存库——
> 看起来一切正常，实际检索的是空索引。`VECTOR_DB_STRICT=true` 能让它直接失败。

---

## 三、启动

### 3.1 本地直跑

```bash
PYTHONPATH=. ./.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

启动时会执行**轻量自检**（探活基础设施，不真实调用 LLM，避免消耗配额与卡启动），
知识库为空时自动完成首次建索引。深度自检（含真实模型问答）访问 `/test/all`。

### 3.2 Docker Compose

```bash
docker compose up -d          # 默认只起 app + redis
docker compose logs -f app
docker compose down           # 停止；加 -v 会连数据卷一起删
```

默认编排**只含应用 + Redis**。需要 Milvus 时用 profile 按需拉起：

```bash
# 构建时必须把向量库客户端装进镜像，否则容器内缺包会降级为内存库
WITH_VECTOR_CLIENTS=1 VECTOR_DB_TYPE=milvus docker compose --profile milvus up -d --build
```

Milvus Standalone 会一并启动所依赖的 **etcd** 与 **minio**，共三个容器，
首次启动约需 60~90s 才健康（`start_period: 90s`）。
把 `WITH_VECTOR_CLIENTS=1`、`VECTOR_DB_TYPE=milvus` 写进 `.env` 后，
后续只需 `docker compose --profile milvus up -d`。

### 3.3 数据持久化目录

以下四个目录必须挂载到宿主机，否则容器重建即丢失：

| 目录 | 内容 | 能否重建 |
|---|---|---|
| `data/` | 企业知识库原始文档 | ❌ 源数据 |
| `vector_store/` | 向量索引 + 词面索引 | ✅ 可重建（需 embedding 配额） |
| `memory_store/` | 长期记忆（用户画像 + 对话原文） | ❌ 不可重建 |
| `logs/` | 运行日志与 trace | ❌ 不可重建 |

---

## 四、部署验证（四步，缺一不可）

很多人只做第 1 步就认为"部署成功"，然后在第一次提问时才发现检索的是空索引。

### 4.1 健康检查

```bash
curl --noproxy '*' http://localhost:8001/health
```

> `--noproxy '*'` 不是可选项：**本机开着代理时，访问 localhost 会走代理并返回 502**，
> 这是本项目踩过的坑。服务器上同样建议显式加上。

返回体中 `startup_check.passed / total` 是启动自检的通过数。

### 4.2 确认向量库没有静默降级

```bash
PYTHONPATH=. ./.venv/bin/python scripts/check_vector_db.py
# 容器内：
docker compose exec app python scripts/check_vector_db.py
```

**这一步不能省。** 它回答的是"现在检索的到底是 Milvus、Chroma，还是内存库"。

### 4.3 建索引

```bash
curl -X POST http://localhost:8001/knowledge/rebuild
```

### 4.4 端到端冒烟

单测覆盖不到"真实模型 + 真实向量库"这一段，必须手工跑一次：

```bash
curl -X POST http://localhost:8001/chat/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"年假有多少天","session_id":"smoke-1","user_id":"admin"}'
```

预期：`answer` 非空，`sources` 里的来源确实来自 `data/` 下的文档。

---

## 五、进程守护与反向代理

### 5.1 systemd

> 以下为**示例**，路径与用户需按实际替换。

```ini
# /etc/systemd/system/enterprise-bot.service
[Unit]
Description=Enterprise RAG Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=appuser
WorkingDirectory=/opt/langgraph-enterprise-bot
EnvironmentFile=/opt/langgraph-enterprise-bot/.env
ExecStart=/opt/langgraph-enterprise-bot/.venv/bin/python -m uvicorn app.main:app \
          --host 127.0.0.1 --port 8001
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now enterprise-bot
sudo journalctl -u enterprise-bot -f
```

> ⚠️ `EnvironmentFile` 的解析规则与 `python-dotenv` **不完全一致**：
> 值里含未加引号的空格、或跨多行的值会被 systemd 拒绝。
> 若 `.env` 有这类内容，改用：
> `ExecStart=/bin/bash -c 'set -a && . /opt/langgraph-enterprise-bot/.env && exec /opt/langgraph-enterprise-bot/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8001'`

> ❌ **不要用 `--workers N`**，理由见第一节。

### 5.2 Nginx 反向代理

```nginx
server {
    listen 443 ssl;
    server_name bot.example.com;

    ssl_certificate     /etc/ssl/certs/bot.crt;
    ssl_certificate_key /etc/ssl/private/bot.key;

    client_max_body_size 20m;      # 知识库上传文档需要

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # SSE 流式对话：必须关缓冲，否则 token 会被 nginx 攒着不发
    location /chat/ask/stream {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_set_header Host $host;
    }
}
```

**两个必须注意的点：**

1. **`proxy_buffering off` 不能省。** `/chat/ask/stream` 是 SSE，
   nginx 默认会缓冲响应体，表现为"前端等了很久，然后一次性吐出全部内容"——
   流式效果完全失效，且不报错。

2. **限流按 IP 计数，而 IP 来自 `X-Real-IP` / `X-Forwarded-For`。**
   当 nginx 与应用**不在同一台机器**时，uvicorn 默认只信任来自 `127.0.0.1`
   的转发头，于是所有用户会被算作同一个 IP，**共享 20 次/分钟的配额**。
   此时需显式指定：

   ```bash
   ... -m uvicorn app.main:app --proxy-headers --forwarded-allow-ips='<nginx 的 IP>'
   ```

免鉴权与免限流的路径：`/static`、`/health`、`/`、`/dify/info`（运维探针，不占配额）。

---

## 六、备份与恢复

### 6.1 先分清哪些能重建

| 目标 | 能重建？ | 说明 |
|---|---|---|
| `data/` | ❌ | 原始文档是唯一真源（若另有文档管理系统则视情况） |
| `memory_store/` | ❌ | 长期记忆是运行中积累的，删了就没了 |
| `.env` | ❌ | 密钥，**不要进版本库** |
| `vector_store/` | ✅ | `POST /knowledge/rebuild` 重建，但需要 embedding 配额与时间 |
| Redis 数据 | ✅ | 缓存性质，丢失只影响性能 |

**优先备份前两项。** `vector_store/` 体积大且可重建，按需决定。

> ⚠️ 仓库里的 `scripts/backup.sh` 是**开发期的文件级快照工具**（配合切分改造使用），
> **不是生产备份方案**——它把文件复制到 `artifacts/backup/` 下，没有保留策略、没有校验。
> 生产请用下面的方式。

### 6.2 备份

```bash
# 停服务后再备（见 6.3 的说明）
docker compose stop app

TS=$(date +%Y%m%d-%H%M%S)
tar czf "backup-$TS.tar.gz" data/ memory_store/ .env

# 索引单独备份（可省，但有它恢复更快）
tar czf "vector_store-$TS.tar.gz" vector_store/

docker compose start app
```

若用 Milvus，还需备份其数据卷（`milvus-data` / `etcd-data` / `minio-data`）。

### 6.3 恢复

```bash
docker compose stop app          # ⚠️ 必须先停：运行中的进程会覆写 vector_store

tar xzf backup-<TS>.tar.gz       # 覆盖 data/ memory_store/ .env
docker compose start app

# 索引若未恢复，重建一次
curl -X POST http://localhost:8001/knowledge/rebuild
```

**为什么必须先停服务**：向量库的写入是"先删后增"的幂等语义，
进程运行中恢复文件会造成内存状态与磁盘状态不一致，且不会有任何报错。

---

## 七、升级与回滚

### 7.1 升级

```bash
# 1) 先备份（见第六节）
# 2) 拉取新代码
git pull

# 3) 跑门禁三连，全绿才继续
PYTHONPATH=. ./.venv/bin/python -m pytest tests/ -q
./.venv/bin/ruff check app scripts tests --no-cache --output-format concise
./.venv/bin/python scripts/verify_doc_linenos.py

# 4) 重建镜像并重启
docker compose up -d --build
```

> 门禁没有 CI 承载（本仓库当前不含 CI 配置），是**本地纪律**，不要省。

### 7.2 回滚

```bash
git log --oneline -5
git revert <bad-commit>          # 推荐：保留历史
# git reset --hard <good-commit> # 仅在确认无人依赖当前提交时使用

docker compose up -d --build
```

> ⚠️ **回滚代码不会回滚索引。** `vector_store/` 在宿主机上，
> 若本次改动涉及切分逻辑（`chunk_documents` 等），回滚后必须
> `POST /knowledge/rebuild` 重建，否则会出现"代码是旧版、索引是新版"的错配。
> 这类错配的典型症状是检索结果与代码行为对不上，且日志一切正常。

---

## 八、故障排查速查

| 症状 | 先查什么 |
|---|---|
| 本机 curl 返回 502 | 是不是走了系统代理？加 `--noproxy '*'` |
| 服务起来了但检索不到东西 | `scripts/check_vector_db.py`——是否静默降级为内存库 |
| 某个问法就是问不出来 | 索引是否丢片段：向量库条数 vs 词面索引条数是否相等 |
| 流式接口一次性返回全部内容 | nginx 的 `proxy_buffering off` 漏了 |
| 所有人共享一个限流配额 | 反代场景下 `--forwarded-allow-ips` 没配 |
| 追问时行为不连贯 | 会话 pin 是否因多副本而失效（见第一节） |
| `/routing/stats` 数字对不上 | 同上；单实例下不会是这个问题 |
| 启动卡住不监听端口 | 启动自检在真实调用 LLM，账号限流期会一直退避重试 |

日志与追踪：`logs/trace.jsonl`（每请求一行）、`logs/` 下的运行日志；
响应里的 `trace_id` 可用于串联一次请求的全部日志。

---

## 附：常用命令

```bash
# 启动 / 停止
docker compose up -d                          # 默认（app + redis）
docker compose --profile milvus up -d         # 含 Milvus 三容器
docker compose down                           # 停止

# 验证
curl --noproxy '*' http://localhost:8001/health
PYTHONPATH=. ./.venv/bin/python scripts/check_vector_db.py

# 索引
curl -X POST http://localhost:8001/knowledge/rebuild
curl http://localhost:8001/knowledge/list

# 门禁三连（无 CI，靠本地纪律）
PYTHONPATH=. ./.venv/bin/python -m pytest tests/ -q
./.venv/bin/ruff check app scripts tests --no-cache --output-format concise
./.venv/bin/python scripts/verify_doc_linenos.py

# 改过代码行数后回填文档行号
./.venv/bin/python scripts/fix_doc_linenos.py          # 先 dry-run
./.venv/bin/python scripts/fix_doc_linenos.py --write
```
