#!/usr/bin/env python3
"""全链路自动化测试脚本 —— 4 阶段 15 项测试。

用法：
    python tests/test_service.py                          # 全量测试
    python tests/test_service.py --url http://host:port   # 指定服务地址
    python tests/test_service.py --skip-heavy             # 跳过耗时项（索引重建等）
    python tests/test_service.py -v                       # 输出详细日志
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import List, Tuple

DEFAULT_URL = "http://localhost:8001"

GREEN, RED, YELLOW, DIM, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"


class Client:
    """极简 HTTP 客户端（不依赖 requests，避免测试环境额外安装）。"""

    def __init__(self, base_url: str, verbose: bool = False):
        self.base = base_url.rstrip("/")
        self.verbose = verbose
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 绕过系统代理

    def request(self, method: str, path: str, body=None, timeout: int = 120):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with self.opener.open(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
            if self.verbose:
                print(f"{DIM}      ← {method} {path} → {json.dumps(payload, ensure_ascii=False)[:220]}{RESET}")
            return resp.status, payload
        except urllib.error.HTTPError as e:
            raw = e.read().decode()
            try:
                return e.code, json.loads(raw)
            except json.JSONDecodeError:
                return e.code, {"detail": raw}

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def post(self, path, body, **kw):
        return self.request("POST", path, body, **kw)

    def delete(self, path, **kw):
        return self.request("DELETE", path, **kw)


class Runner:
    def __init__(self, client: Client, skip_heavy: bool):
        self.c = client
        self.skip_heavy = skip_heavy
        self.results: List[Tuple[str, str, bool, str, int]] = []

    def check(self, stage: str, name: str, fn, heavy: bool = False):
        if heavy and self.skip_heavy:
            print(f"  {YELLOW}SKIP{RESET} {name}（已跳过耗时项）")
            return
        t0 = time.perf_counter()
        try:
            detail = fn() or "OK"
            ok, ms = True, int((time.perf_counter() - t0) * 1000)
            print(f"  {GREEN}PASS{RESET} {name} {DIM}· {detail} · {ms}ms{RESET}")
        except AssertionError as e:
            ok, detail, ms = False, str(e), int((time.perf_counter() - t0) * 1000)
            print(f"  {RED}FAIL{RESET} {name} {DIM}· {detail} · {ms}ms{RESET}")
        except Exception as e:  # noqa: BLE001
            ok, detail, ms = False, f"{type(e).__name__}: {e}", int((time.perf_counter() - t0) * 1000)
            print(f"  {RED}FAIL{RESET} {name} {DIM}· {detail} · {ms}ms{RESET}")
        self.results.append((stage, name, ok, detail, ms))

    def summary(self) -> int:
        total, passed = len(self.results), sum(1 for r in self.results if r[2])
        print("\n" + "=" * 62)
        for stage in ["1 服务健康", "2 知识库", "3 工作流", "4 对话链路"]:
            items = [r for r in self.results if r[0] == stage]
            if not items:
                continue
            ok = sum(1 for r in items if r[2])
            color = GREEN if ok == len(items) else RED
            print(f"  {color}{stage}{RESET}: {ok}/{len(items)} 通过")
        print("=" * 62)
        color = GREEN if passed == total else RED
        print(f"  {color}总计 {passed}/{total} 通过{RESET} · 平均耗时 "
              f"{sum(r[4] for r in self.results) // max(total, 1)}ms")
        print("=" * 62)
        return 0 if passed == total else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="企业智能助手全链路测试")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"服务地址（默认 {DEFAULT_URL}）")
    parser.add_argument("--skip-heavy", action="store_true", help="跳过耗时测试项")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出详细响应")
    args = parser.parse_args()

    client, runner = Client(args.url, args.verbose), None
    runner = Runner(client, args.skip_heavy)

    print(f"\n企业智能助手 · 全链路测试\n目标服务：{args.url}\n")

    # ---------------- 阶段 1：服务健康 ----------------
    print("[阶段 1] 服务健康")
    state = {"session": None}

    def t_health():
        code, d = client.get("/health")
        assert code == 200, f"HTTP {code}"
        assert d.get("code") == 0, "响应 code 非 0"
        return f"llm={d.get('llm_mode')} vector={d.get('vector_db')} cache={d.get('cache')}"

    def t_quick():
        code, d = client.get("/test/quick")
        assert code == 200, f"HTTP {code}"
        assert d.get("success"), f"快速检测未全部通过：{d.get('passed')}/{d.get('total')}"
        return f"通过 {d['passed']}/{d['total']}"

    def t_openapi():
        code, d = client.get("/openapi.json")
        assert code == 200, f"HTTP {code}"
        paths = d.get("paths", {})
        assert len(paths) >= 10, f"接口数量异常：{len(paths)}"
        return f"{len(paths)} 个接口"

    runner.check("1 服务健康", "健康检查 /health", t_health)
    runner.check("1 服务健康", "快速自测 /test/quick", t_quick)
    runner.check("1 服务健康", "OpenAPI 接口清单", t_openapi)

    # ---------------- 阶段 2：知识库 ----------------
    print("\n[阶段 2] 知识库管理")

    def t_kb_list():
        code, d = client.get("/knowledge/list")
        assert code == 200, f"HTTP {code}"
        assert d.get("total_chunks", 0) > 0, "知识库为空"
        return f"{d['total_chunks']} 条片段 / {len(d['sources'])} 篇文档"

    def t_kb_search():
        code, d = client.post("/knowledge/search", {"query": "年假有多少天", "top_k": 3})
        assert code == 200, f"HTTP {code}"
        assert d.get("hits"), "检索无结果"
        return f"Top1={d['hits'][0]['score']} 来源={d['hits'][0]['source']}"

    def t_kb_upload():
        code, d = client.post(
            "/knowledge/upload",
            {"file_name": "__test_doc.txt", "content": "测试文档：星河科技年度团建定于每年 11 月举行，费用由公司全额承担。"},
        )
        assert code == 200, f"HTTP {code} {d}"
        assert d.get("chunks", 0) > 0, "入库切片数为 0"
        return f"入库 {d['chunks']} 条"

    def t_kb_delete():
        code, d = client.delete("/knowledge/__test_doc.txt")
        assert code == 200, f"HTTP {code} {d}"
        return f"移除 {d.get('removed', 0)} 条"

    def t_kb_rebuild():
        code, d = client.post("/knowledge/rebuild", {}, timeout=300)
        assert code == 200, f"HTTP {code} {d}"
        assert d.get("chunks", 0) > 0, "重建后片段数为 0"
        return f"{d['documents']} 篇 → {d['chunks']} 条"

    runner.check("2 知识库", "文档列表 /knowledge/list", t_kb_list)
    runner.check("2 知识库", "语义检索 /knowledge/search", t_kb_search)
    runner.check("2 知识库", "文档上传 /knowledge/upload", t_kb_upload)
    runner.check("2 知识库", "文档删除 /knowledge/{name}", t_kb_delete)
    runner.check("2 知识库", "索引重建 /knowledge/rebuild", t_kb_rebuild, heavy=True)

    # ---------------- 阶段 3：工作流 ----------------
    print("\n[阶段 3] 工作流引擎")

    def t_wf_status():
        code, d = client.get("/workflow/status")
        assert code == 200, f"HTTP {code}"
        assert len(d.get("nodes", [])) == 5, f"节点数异常：{len(d.get('nodes', []))}"
        return f"{len(d['nodes'])} 节点 / {len(d['branches'])} 分支"

    def t_wf_exec():
        code, d = client.post("/workflow/execute", {"query": "VPN 连接不上怎么办"})
        assert code == 200, f"HTTP {code} {d}"
        assert d.get("answer"), "未生成答案"
        return f"intent={d.get('intent')} · {len(d.get('trace', []))} 节点"

    runner.check("3 工作流", "工作流状态 /workflow/status", t_wf_status)
    runner.check("3 工作流", "手动执行 /workflow/execute", t_wf_exec)

    # ---------------- 阶段 4：对话链路 ----------------
    print("\n[阶段 4] 对话链路")

    def t_chat_knowledge():
        code, d = client.post("/chat/ask", {"query": "年假有多少天？"})
        assert code == 200, f"HTTP {code} {d}"
        assert d.get("answer"), "未返回答案"
        assert d.get("sources"), "未返回检索来源"
        state["session"] = d.get("session_id")
        return f"intent={d['intent']} · {len(d['sources'])} 条来源 · {d['elapsed_ms']}ms"

    def t_chat_multi():
        sid = state.get("session")
        assert sid, "前置会话未建立"
        code, d = client.post("/chat/ask", {"query": "那病假需要提供什么证明？", "session_id": sid})
        assert code == 200, f"HTTP {code} {d}"
        assert d.get("history_rounds", 1) >= 2, f"未继承上下文：{d.get('history_rounds')} 轮"
        return f"继承 {d['history_rounds']} 轮上下文"

    def t_chat_tool():
        code, d = client.post("/chat/ask", {"query": "E1001 是哪个部门的？"})
        assert code == 200, f"HTTP {code} {d}"
        assert d.get("intent") == "tool", f"意图判定错误：{d.get('intent')}"
        assert "技术部" in (d.get("answer") or ""), "回答未包含员工信息"
        return f"工具调用成功 · {d['elapsed_ms']}ms"

    def t_chat_guard():
        code, d = client.post("/chat/ask", {"query": "<script>alert(1)</script>年假"})
        assert code == 200, f"HTTP {code} {d}"
        assert "<script>" not in (d.get("answer") or ""), "注入内容未被过滤"
        return "注入过滤生效"

    def t_chat_clear():
        sid = state.get("session")
        assert sid, "前置会话未建立"
        code, _ = client.delete(f"/chat/history/{sid}")
        assert code == 200, f"HTTP {code}"
        return "会话已清空"

    runner.check("4 对话链路", "知识问答", t_chat_knowledge)
    runner.check("4 对话链路", "多轮上下文继承", t_chat_multi)
    runner.check("4 对话链路", "业务工具调用（员工信息）", t_chat_tool)
    runner.check("4 对话链路", "注入安全防护", t_chat_guard)
    runner.check("4 对话链路", "会话历史清理", t_chat_clear)

    return runner.summary()


if __name__ == "__main__":
    sys.exit(main())
