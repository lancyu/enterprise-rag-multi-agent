"""API Key 的**解析与比较** —— 全项目唯一实现处。

为什么必须只有一处
------------------
入站鉴权中间件（``app/main.py``）与 Dify 兼容端点（``app/api/dify.py``）各写过一套，
两套规则**不同且判定相反**：

============================  ==============================  ==================
输入                          入站中间件                       Dify 端点
============================  ==============================  ==================
``Bearer xxx``                通过                            通过
``Bearer    xxx``（多个空格）  通过                            通过
``Bearer\\txxx``（制表符）      **拒绝**                        **通过**
``X-API-Key: xxx``             通过                            **不认**（只读 Authorization）
============================  ==============================  ==================

同一个请求在两条入口上得到相反结论，**两边都不报错**，只表现为
"用同一个 token，走 A 端点 200、走 B 端点 401"。这类分歧没有任何日志会指向它，
而它恰好发生在**安全边界**上。

规则取更严格的一侧（也就是 RFC 6750 的写法：``Bearer`` + 一个或多个 **空格** + 令牌）。
刻意不用 ``str.split()``——它按**任意空白**切，会把制表符、换行也算成合法分隔符，
于是"格式更宽松"这件事会以"两边不一致"的形式重新出现。
"""
from __future__ import annotations

import secrets
from typing import Optional

#: 前缀（含其后的空格）。比较时统一转小写：RFC 7235 规定 scheme 不区分大小写。
_SCHEME = "bearer "


def bearer_token(authorization: Optional[str]) -> str:
    """从 ``Authorization`` 头里取出 Bearer 令牌；格式不合法返回空串。

    只认 ``Bearer <空格><令牌>``。返回空串对调用方只有一种含义：
    **这个头没有给出可用的令牌**——至于是"头缺失"还是"格式不对"，
    由调用方自己看原始头决定（Dify 端点要据此回不同的错误码）。
    """
    value = (authorization or "").strip()
    if not value.lower().startswith(_SCHEME):
        return ""
    return value[len(_SCHEME):].strip()


def api_key_matches(provided: str, expected: str) -> bool:
    """定长安全比较，并**把 fail-closed 写在这里**。

    两处都用 ``secrets.compare_digest`` 时，最容易出的错是漏掉其中一处——
    于是某条入口退化成可被计时攻击的普通字符串比较，而**没有任何测试会变红**。

    ``expected`` 为空 → 一律不通过：配了鉴权却没配密钥时，"谁都能过"
    比"谁都不过"危险得多，因为管理员会误以为已经防护上了。
    """
    if not expected:
        return False
    return secrets.compare_digest(provided, expected)
