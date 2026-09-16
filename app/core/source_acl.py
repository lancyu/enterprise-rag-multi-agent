"""按用户的来源访问控制（Source ACL）。

把「用户身份」解析为「允许访问的文档来源白名单」，使 `retrieve()` 的
`allowed_sources` 参数真正落地为端到端的知识隔离（承接 docs/history/project-assessment.md
对「retrieve 加权限参数」的建议）。

设计原则（最小权限 + fail-closed）
---------------------------------
- ACL 未配置（空 dict）→ 不限制（返回 ``None``），与无 ACL 部署行为一致。
- 配置了 ACL 且能匹配到具体用户 → 使用该用户的来源模式列表。
- 配置了 ACL 但未匹配到具体用户：
    - 若存在 ``"*"`` 默认规则，则套用其列表；
    - 否则返回 ``[]``（全部拒绝）—— 宁可让用户拿不到答案，也不越权泄露。
- 来源模式列表中含 ``"*"`` → 表示允许全部来源（返回 ``None``，不限制）。

返回的列表元素即 fnmatch 模式（如 ``"hr/*"``、``"finance/report.xlsx"``），
由 `retrieve()` 的 `filter_by_allowed_sources` 做通配匹配。
"""
from typing import Dict, List, Optional

from app import config


def resolve_allowed_sources(user_id: Optional[str]) -> Optional[List[str]]:
    """把 user_id 解析为允许访问的来源模式列表。

    Returns:
        ``None`` 表示不限制（放行全部来源）；``[]`` 表示全部拒绝；
        其余为 fnmatch 来源模式列表。
    """
    acl: Dict[str, List[str]] = getattr(config, "SOURCE_ACL", None) or {}
    if not acl:
        return None

    key = user_id or "default"
    patterns = acl.get(key)
    if patterns is None:
        patterns = acl.get("*")

    if patterns is None:
        # 配了 ACL 但既无该用户、也无默认规则 → fail-closed 全部拒绝
        return []

    if "*" in patterns:
        # 显式通配 → 不限制
        return None

    return list(patterns)
