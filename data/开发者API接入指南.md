# 星河智能云平台 · 开发者 API 接入指南

> 版本：V1.0 | 更新日期：2026-07-25
> 适用对象：需要在自有系统中集成星河智能云平台能力的开发者

本指南介绍如何通过标准 REST API 接入平台的模型调用、知识库检索与智能体编排能力。所有接口通过统一网关调度，采用 API Key 鉴权，支持按业务场景路由不同模型。

## 1. 鉴权与网关

平台所有接口均通过 HTTPS 调用，请求头须携带 `Authorization: Bearer <API_KEY>`。API Key 在控制台"模型服务密钥"页创建，可按应用维度隔离与轮换。建议将密钥存入环境变量，避免硬编码在代码中。

```bash
curl -X POST "https://api.company.com/v1/chat/completions" \
  -H "Authorization: Bearer $XINGHE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"kimi-k2.6","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

### 1.1 网关限流

统一网关默认按 API Key 维度限流，免费额度为每分钟 3 次请求（RPM=3），超出后返回 429 并建议稍后重试。生产环境可申请提升配额。客户端应实现指数退避重试，避免请求被直接丢弃。

### 1.2 区域与端点

公有云端点为 `api.company.com`；私有化部署端点由交付文档提供，通常为内网域名。混合部署下，编排请求走云端控制面，数据面请求直连客户本地网关。

## 2. 模型调用

文本生成接口兼容 OpenAI Chat Completions 协议，便于已有代码平滑迁移。支持 `stream` 流式返回，首包延迟（TTFT）受模型与网络影响。

### 2.1 请求参数

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| model | string | 模型标识，如 kimi-k2.6 |
| messages | array | 对话消息，含 system/user/assistant |
| temperature | float | 采样温度，默认 1.0 |
| max_tokens | int | 最大生成长度，默认 1024 |
| stream | bool | 是否流式，建议开启以优化体验 |

### 2.2 流式解析

流式响应以 `data: {JSON}` 分行推送，结束时发送 `data: [DONE]`。客户端需逐行解析 content 增量并拼接。若流式全程无有效内容，应降级为一次性 invoke 调用，避免返回空回答。

```python
import httpx, json

def stream_answer(prompt: str):
    with httpx.Client(timeout=60) as c:
        with c.stream("POST", f"{BASE}/v1/chat/completions",
                      headers={"Authorization": f"Bearer {KEY}"},
                      json={"model": "kimi-k2.6", "messages": [{"role": "user", "content": prompt}], "stream": True}) as r:
            for line in r.iter_lines():
                if line.startswith("data: ") and "[DONE]" not in line:
                    yield json.loads(line[6:])["choices"][0]["delta"].get("content", "")
```

## 3. 知识库检索

知识库接口提供文档上传、增量解析与语义检索。检索返回 Top-K 片段及相似度分数，可叠加重排序模型提升精度。

### 3.1 上传与解析

文档上传支持 PDF、Word、Markdown、TXT、HTML，单次上限 50 MB。内容修改后调用"重新解析"增量更新，无需重新上传。建议切片控制在 300 至 500 字符，并为文档补充标题与摘要以提升语义表征。

### 3.2 检索调用

```bash
curl -X POST "https://api.company.com/v1/kb/retrieve" \
  -H "Authorization: Bearer $XINGHE_API_KEY" \
  -d '{"kb_id":"kb_123","query":"年假有多少天","top_k":5}'
```

返回结果包含候选片段文本、来源文档与融合得分。融合得分由稠密召回与词面召回经 RRF 算法融合归一化得到，可跨模型比较。

## 4. 智能体编排

平台支持通过 API 创建与发布智能体，将检索、模型调用与业务工具编排为可复用流程。编排定义以 JSON 描述节点与边，支持条件分支与并行。

### 4.1 创建智能体

调用 `POST /v1/agents` 提交编排定义，返回 agent_id。定义中包含入口节点、检索节点、生成节点与输出节点。节点间通过状态对象传递上下文。

### 4.2 调用与追踪

智能体运行后可获取结构化追踪（trace），包含各节点耗时与属性，便于定位瓶颈。首 token 延迟（TTFT）与总耗时分别记录，可区分网络排队与模型生成慢。

## 5. 错误处理

接口统一返回业务码与错误消息。429 表示限流，500 表示服务端异常。客户端应对 429 实现退避重试，对 5xx 做有限次重试，对 4xx 业务错误直接暴露以便排查。所有请求建议携带请求 ID 以便于工单追溯。

## 6. 知识库管理 API

知识库接口支持文档的增删改查与增量解析。上传后平台自动完成解析、切片与向量化，检索前无需人工干预。

### 6.1 创建知识库

调用 `POST /v1/kb` 创建知识库，返回 `kb_id`。可按业务域设置切片参数与检索权重，参数变更对所有文档生效。

### 6.2 文档管理

| 操作 | 方法 | 说明 |
| --- | --- | --- |
| 上传文档 | POST /v1/kb/{kb_id}/documents | 支持 PDF、Word、Markdown、TXT、HTML，单次 50 MB |
| 重新解析 | POST /v1/kb/{kb_id}/documents/{doc_id}/reparse | 内容修改后增量更新，无需重新上传 |
| 删除文档 | DELETE /v1/kb/{kb_id}/documents/{doc_id} | 同步清理向量与词面索引 |
| 重建索引 | POST /v1/kb/{kb_id}/reindex | 全量刷新，期间检索不中断 |

## 7. 智能体 API

### 7.1 创建与发布

调用 `POST /v1/agents` 提交编排定义（JSON 描述节点与边），返回 `agent_id`；调用 `POST /v1/agents/{agent_id}/publish` 发布为可用端点。编排支持条件分支、并行与人工介入节点。

### 7.2 运行与追踪

运行智能体返回结构化 trace，包含各节点耗时与属性，便于定位瓶颈。首 token 延迟（TTFT）与总耗时分别记录，可区分网络排队与模型生成慢。

## 8. Webhook 与事件

平台支持将关键事件（如索引完成、调用异常、额度告警）通过 Webhook 推送到客户系统。回调体含 `event_type`、`timestamp` 与 `payload`，建议按 `event_type` 分流处理并做幂等去重。

## 9. SDK 与最佳实践

官方提供 Python 与 JavaScript SDK，封装鉴权、限流退避与流式解析。生产环境建议：密钥存环境变量、统一异常包装、为长请求设置超时与取消、对 429 实现指数退避（基数 1s、上限 5s、加随机抖动）。所有请求携带请求 ID 以便于工单追溯。
