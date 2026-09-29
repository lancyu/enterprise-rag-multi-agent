#!/usr/bin/env python3
"""`artifacts/backup/` 的保留策略 —— 只清理**自动生成**的快照。

为什么需要它
------------
`fix_doc_linenos.py --write` 每回填一次就留一份快照。留快照本身是对的
（回填行号是批量改写，需要能回退），但它**没有上界**。

代价不是磁盘，是**搜索**。2026-09-29 实测：`artifacts/backup/` 里 **67 份自动快照
装的全是 `project-introduction.md`**，各是同一个文件在不同日期的版本。于是：

    grep -rn "工具链路不经校验" .        # 会命中一堆早就修好的旧副本

同一句话在"现行文档"和"某天的快照"里各有一份，快照那份说的是**当时**的事实。
它会让筛查的人（以及静态扫描、全文检索）反复命中过期实现——
分辨"这是现行实现"还是"这是某天的副本"要额外花力气，而**漏掉这个分辨就会照着错的下结论**。
`scripts/backup.sh` 的注释里早就写下了同一条教训：

> 旧版源码副本留在仓库里会让静态扫描与全文检索反复命中过期实现，
> 是审查时最容易踩的坑。

保留策略
--------
- **只动** `doc-linenos-<8位日期>-<6位时间>` 这种**脚本亲手写出来**的目录名；
- **不动** `backup.sh` 按任务 id 建的手工快照——那些装多个文件、是人的决策产物，
  按数量自动丢弃它们等于替人做决定；
- 按时间戳倒序保留最近 `--keep` 份，其余**移入废纸篓**（`mv`，不是 `rm`）。

⚠️ **判据必须严格到"日期-时间"这个形状，不能图省事按 `doc-linenos-` 前缀匹配。**
目录里真有一个 `doc-linenos-mine-230057`：名字借用了同一个前缀，但里面装的是
**4 份人工挑的文档**（`feasibility-and-value.md` / `value-remediation-plan.md` /
`tool-invocation-online-vs-offline.md` / `intent-routing-hardening-plan.md`）。
按前缀匹配会把它连同内容一起静默丢掉——不会报错，只是某天想找这几份文档时发现没了。
判据能定得这么死，是因为自动快照的名字**由本项目的脚本自己生成**；
换成一个"什么名字都可能有"的目录，就该反过来放宽并让人工确认。

用法::

    python scripts/prune_backups.py                 # dry-run：只列出来
    python scripts/prune_backups.py --apply         # 真移
    python scripts/prune_backups.py --apply --keep 3
"""
from __future__ import annotations

import argparse
import pathlib
import re
import shutil
from datetime import datetime

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: 自动快照的名字形状（由 `fix_doc_linenos.py` 生成）。
AUTO_NAME_RE = re.compile(r"^doc-linenos-(\d{8})-(\d{6})$")

#: 默认保留份数。取 5 是因为回填失败通常当场就发现，不需要一个月的历史。
DEFAULT_KEEP = 5


def auto_snapshots(root: pathlib.Path) -> list:
    """自动快照目录，**新的在前**。手工快照（名字不匹配）一律不在结果里。"""
    if not root.is_dir():
        return []
    found = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = AUTO_NAME_RE.match(child.name)
        if match:
            found.append((f"{match.group(1)}-{match.group(2)}", child))
    return [path for _, path in sorted(found, reverse=True)]


def plan(root: pathlib.Path, keep: int = DEFAULT_KEEP):
    """返回 ``(要保留的, 要清掉的)`` —— 纯函数，不做任何写操作。

    ``keep < 1`` 直接抛错而不是"清空"：清空等于把回退能力也一起丢掉，
    那不是这个脚本的职责，多半是调用方写错了参数。**上界判定只有这一处**。
    """
    if keep < 1:
        raise ValueError("--keep 至少要 1（0 会把回退能力也一起丢掉）")
    shots = auto_snapshots(root)
    return shots[:keep], shots[keep:]


def prune(
    root: pathlib.Path,
    keep: int = DEFAULT_KEEP,
    apply: bool = False,
    trash_root: pathlib.Path | None = None,
):
    """执行保留策略，返回 ``(保留, 清掉, 废纸篓目录 or None)``。

    ``apply=False``（默认）只做计划，**不碰文件系统**：调用方可以先看清单。
    真移时一律进废纸篓——清错了还能捞回来，这正是 `rm` 与 `mv` 的差别。

    ``trash_root`` 存在的唯一理由是**测试**：不给出这个参数，测试就会往真实的
    ``~/.Trash`` 里扔东西，跑一次留一堆垃圾。生产调用一律用默认值。
    """
    kept, dropped = plan(root, keep)
    if not apply or not dropped:
        return kept, dropped, None

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = trash_root if trash_root is not None else pathlib.Path.home() / ".Trash"
    dest_root = base / f"prune-backups-{stamp}"
    dest_root.mkdir(parents=True, exist_ok=True)
    for path in dropped:
        target = dest_root / path.name
        if target.exists():                       # 同一秒内跑两次也不会互相覆盖
            target = dest_root / f"{path.name}-{stamp}"
        shutil.move(str(path), str(target))
    return kept, dropped, dest_root


def main() -> int:
    parser = argparse.ArgumentParser(
        description="清理 artifacts/backup/ 里自动生成的文档快照（只留最近 N 份）"
    )
    parser.add_argument("--apply", action="store_true", help="真的移动；不加则只列出")
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP, help=f"保留份数（默认 {DEFAULT_KEEP}）")
    parser.add_argument("--root", default=str(REPO_ROOT / "artifacts" / "backup"))
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    try:
        kept, dropped, dest = prune(root, keep=args.keep, apply=args.apply)
    except ValueError as exc:                      # 上界判定在 plan() 里，别在这儿再写一遍
        print(exc)
        return 2

    print(f"目录：{root}")
    print(f"自动快照：保留 {len(kept)} 份，清理 {len(dropped)} 份")
    for path in kept:
        print(f"  保留  {path.name}")
    for path in dropped:
        print(f"  {'已移' if args.apply else '待移'}  {path.name}")

    if not dropped:
        print("\n已在上界之内，无需清理。")
    elif args.apply and dest:
        print(f"\n已移入废纸篓：{dest}")
        print("确认无误后可在访达里清空；想反悔就把它移回原处。")
    else:
        print("\n（dry-run，未移动。加 --apply 生效）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
