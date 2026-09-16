# ============================================================
# 企业智能助手 · RAG 知识引擎 镜像
# 构建：docker build -t enterprise-bot:1.0.0 .
# 运行：docker run -d -p 8001:8001 --env-file .env enterprise-bot:1.0.0
# ============================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 系统依赖（编译类依赖按需打开注释）
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，利用 Docker 层缓存
# WITH_VECTOR_CLIENTS：是否把向量库客户端（chromadb / pymilvus）装进镜像。
#   memory 后端（默认）不需要 → 默认 0，镜像保持精简；
#   要用 chroma / milvus → 构建时传 --build-arg WITH_VECTOR_CLIENTS=1，
#   否则容器内会因缺包而降级为内存库（配合 VECTOR_DB_STRICT=true 可让它直接失败）。
ARG WITH_VECTOR_CLIENTS=0
COPY requirements.txt requirements-vector.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt \
    && if [ "$WITH_VECTOR_CLIENTS" = "1" ]; then pip install -r requirements-vector.txt; fi

# 再拷贝源码
COPY . .

# 数据、索引、日志目录（建议运行时挂载宿主机卷）
RUN mkdir -p /app/data /app/vector_store /app/logs

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
