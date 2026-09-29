"""入口脚本的「文档 ↔ 实际接受」必须一致。

为什么要守这个
--------------
`scripts/start.sh` 是**部署期唯一的入口**，它的参数写在 README 的快速开始里。
参数如果漂了，失败方式很安静：照文档敲 `./scripts/start.sh --check`，
脚本回一句"未知参数"就退出——用户会以为是自己敲错了，而不是文档过期了。

这和 `tests/test_topology_rendering.py` 守的是同一类问题：**同一件事在两处各写一遍**
（一处是 README 里的用法，一处是脚本的 `case` 分支），两处会分叉而没有任何报错。
区别只是这里分叉的代价是"装不起来"，而不是"读错架构"。

本文件不真跑脚本（那要建 venv、装依赖、起服务），只做三件离线可判的事：
① 脚本存在、可执行、语法正确；② README 提到的每个参数脚本都认；
③ 帮助入口真的能打印出用法（`usage()` 的锚点没被改坏）。
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

START_SH = REPO_ROOT / "scripts" / "start.sh"
README = REPO_ROOT / "README.md"

#: 参数形如 `--port` / `--no-seed`（只认长选项，短选项 `-h` 不写进 README）。
_FLAG_RE = re.compile(r"(?<![\w-])--[a-z][a-z-]*")


def _script_text() -> str:
    assert START_SH.is_file(), f"{START_SH} 不存在 —— 一键启动入口没了"
    return START_SH.read_text(encoding="utf-8")


def test_script_exists_and_is_executable():
    """必须可执行：README 写的是 `./scripts/start.sh`，不是 `bash scripts/start.sh`。"""
    assert START_SH.is_file(), "scripts/start.sh 不存在"
    assert START_SH.stat().st_mode & 0o111, (
        "scripts/start.sh 没有可执行位 —— README 里的 `./scripts/start.sh` 会失败。"
        "修：chmod +x scripts/start.sh"
    )


def test_script_passes_bash_syntax_check():
    """`bash -n` 只解析不执行 —— 语法错就别等到部署那天才发现。"""
    bash = shutil.which("bash")
    assert bash, "找不到 bash（本仓库的启动脚本是 bash 写的）"
    proc = subprocess.run(
        [bash, "-n", str(START_SH)], capture_output=True, text=True
    )
    assert proc.returncode == 0, f"bash -n 报错：\n{proc.stderr}"


def test_every_flag_documented_in_readme_is_accepted_by_the_script():
    """README 里出现的每个 `--flag` 都必须是脚本认得的参数。

    反面（脚本多支持几个没写进文档的参数）不报——多出来的不算缺陷，
    报错（照文档敲却报"未知参数"）才是。
    """
    script = _script_text()
    # 脚本的 case 分支：`--port)` / `--check)` / `-h|--help)`
    accepted = set(re.findall(r"^\s+(--[a-z][a-z-]*)\)", script, re.M))

    readme = README.read_text(encoding="utf-8")
    block = re.search(r".*start\.sh.*(?:\n.*){0,20}", readme)
    assert block, "README 里找不到 start.sh 的用法说明"

    documented = set()
    for line in readme.split("\n"):
        if "start.sh" in line:
            documented |= set(_FLAG_RE.findall(line))

    assert documented, "README 里没写 start.sh 的任何参数 —— 判据失配，别静默通过"
    unknown = sorted(documented - accepted)
    assert not unknown, (
        f"README 写了这些参数，但 scripts/start.sh 不认：{unknown}\n"
        f"脚本实际接受：{sorted(accepted)}\n"
        f"照文档敲会得到「未知参数」——用户会以为是自己敲错了。"
    )


def test_help_prints_usage():
    """`--help` 必须真的打印用法。

    `usage()` 靠锚点截取自身注释（`# 用法` … `# 退出码`）。锚点被改坏时
    `sed` 会安静地输出空串 —— 帮助还在"成功退出"，只是什么都没说。
    """
    proc = subprocess.run(
        [str(START_SH), "--help"], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, f"--help 退出码 {proc.returncode}：{proc.stderr}"
    out = proc.stdout
    assert "用法" in out, f"--help 没打印用法段（usage() 的锚点可能被改坏了）：{out!r}"
    assert "./scripts/start.sh --check" in out, (
        f"--help 输出里没有关键例子，usage() 可能截错了范围：{out!r}"
    )
