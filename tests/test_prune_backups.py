"""快照保留策略必须真的收敛，且只碰它该碰的东西。

为什么要守这个
--------------
`fix_doc_linenos.py --write` 每回填一次就留一份文档快照，**没有上界**：
2026-09-29 实测 `artifacts/backup/` 已有 74 份 / 30MB，每份装着同一个文件的过期版本。

代价不是磁盘，是**搜索**——筛查问题时 `grep` 会反复命中早就修好的旧副本，
让人以为问题还在（或者以为早就改了）。`scripts/backup.sh` 的注释里已经写过这条教训。

所以这里守的不是"清理能跑通"，而是两条更容易悄悄失效的性质：

1. **只动自动快照**。`backup.sh` 按任务 id 建的手工快照装多个文件、是人的决策产物，
   按数量自动丢弃它们等于替人做决定。判据用"名字形状"而不是"目录里的文件数"，
   因为后者会随着人往里放东西而改变。
2. **默认不动文件系统**。dry-run 必须真的不碰盘——否则"先看一眼"就成了"已经清了"。

反向验证（本项目纪律：修完必须退回修复、确认护栏变红）：
把 `AUTO_NAME_RE` 放宽成 `doc-linenos--` 之类，`test_only_automatic_snapshots_are_candidates`
会红；把 `prune()` 的 `apply` 判断去掉，`test_a_dry_run_touches_nothing` 会红。
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fix_doc_linenos  # noqa: E402  —— 需先把 scripts/ 放进 sys.path
import prune_backups  # noqa: E402

BACKUP_ROOT = REPO_ROOT / "artifacts" / "backup"

#: 自动快照的名字（由 fix_doc_linenos 生成）与手工快照的名字（由 backup.sh 生成）各造几个。
AUTO_NAMES = [
    "doc-linenos-20260901-090000",
    "doc-linenos-20260910-120000",
    "doc-linenos-20260914-190002",
    "doc-linenos-20260920-080000",
    "doc-linenos-20260925-230000",
    "doc-linenos-20260928-110000",
    "doc-linenos-20260929-135045",
]
#: 手工快照：`backup.sh` 按任务 id 建的，以及**借用了 `doc-linenos-` 前缀的手工备份**。
#: 最后那个 `doc-linenos-mine-230057` 是从真实目录里抄来的——它里面装的是 4 份人工挑的文档，
#: 名字却和自动快照同前缀。它是"判据不能按前缀匹配"这条设计的**真实反例**，必须留在样本里。
MANUAL_NAMES = [
    "README.md",
    "p2-8-backup",
    "before-refactor",
    "2026-09-19",
    "doc-linenos-mine-230057",
]


def _newest_first(names: list[str]) -> list[str]:
    """把一组快照名按 `auto_snapshots()` 的返回契约排列（**新的在前**）。

    单独抽出来是因为这个顺序很容易读错：`AUTO_NAMES[-3:]` 看着像"最新的三份"，
    但它是**升序**切片，与返回顺序正好相反。写这个 helper 就是逼调用方想清楚。
    """
    return sorted(names, reverse=True)


def _make_tree(root: Path, auto: list[str] | None = None, manual: list[str] | None = None) -> None:
    for name in auto if auto is not None else AUTO_NAMES:
        (root / name).mkdir(parents=True)
        (root / name / "README.md").write_text("过期副本\n", encoding="utf-8")
    for name in manual if manual is not None else MANUAL_NAMES:
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "app.py").write_text("手工备份\n", encoding="utf-8")


# --------------------------------------------------------------------------- 1. 只动自动快照
def test_only_automatic_snapshots_are_candidates(tmp_path):
    """手工快照一个都不能进候选——按数量自动丢它们是替人做决定。"""
    _make_tree(tmp_path)
    names = {p.name for p in prune_backups.auto_snapshots(tmp_path)}

    assert names == set(AUTO_NAMES), (
        f"候选集不对。多出来的：{names - set(AUTO_NAMES)}；漏掉的：{set(AUTO_NAMES) - names}"
    )
    for manual in MANUAL_NAMES:
        assert manual not in names, f"手工快照 {manual!r} 混进了候选集，会被自动丢掉"


def test_snapshots_are_listed_newest_first(tmp_path):
    """顺序本身是被依赖的契约（`plan()` 用 `[:keep]` 取前 N 份），所以单独钉一条。

    钉它的理由很实在：写这个测试时我自己就把 `AUTO_NAMES[-3:]`（升序切片）
    当成了返回顺序，结果断言反了。契约不写清楚，下一个人还会踩。
    """
    _make_tree(tmp_path)
    listed = [p.name for p in prune_backups.auto_snapshots(tmp_path)]

    assert listed == _newest_first(AUTO_NAMES), (
        f"应从新到旧：期望 {_newest_first(AUTO_NAMES)}，实际 {listed}"
    )


def test_manual_snapshots_survive_even_when_they_outnumber_the_automatic_ones(tmp_path):
    """手工快照数量占多数时也不该被"按时间倒序"顺手带走。"""
    _make_tree(tmp_path, auto=AUTO_NAMES[:2], manual=[f"task-{i:02d}" for i in range(20)])
    kept, dropped, _ = prune_backups.prune(tmp_path, keep=1, apply=True, trash_root=tmp_path / "trash")

    assert [p.name for p in kept] == _newest_first(AUTO_NAMES[:2])[:1] == [AUTO_NAMES[1]]
    assert [p.name for p in dropped] == [AUTO_NAMES[0]]
    for i in range(20):
        assert (tmp_path / f"task-{i:02d}").is_dir(), "手工快照被清掉了"


def test_a_handmade_snapshot_wearing_the_auto_prefix_is_not_touched(tmp_path):
    """名字带 `doc-linenos-` 前缀但**不是**"日期-时间"形状的，一律不算自动快照。

    这条不是假想的边界情况：真实目录里就有一个 `doc-linenos-mine-230057`，
    里面装的是 4 份人工挑的文档（`feasibility-and-value.md` 等）。
    如果判据图省事写成 `name.startswith("doc-linenos-")`，它会连同那 4 份文档
    一起被静默移走——不报错，只是某天想找它们时发现没了。
    """
    _make_tree(tmp_path, auto=AUTO_NAMES[:1], manual=[])
    handmade = tmp_path / "doc-linenos-mine-230057"
    handmade.mkdir()
    (handmade / "feasibility-and-value.md").write_text("人工挑的\n", encoding="utf-8")

    kept, dropped, _ = prune_backups.prune(tmp_path, keep=1, apply=True, trash_root=tmp_path / "trash")

    assert [p.name for p in kept] == [AUTO_NAMES[0]]
    assert dropped == [], "唯一的自动快照只有 1 份，不该清任何东西"
    assert (handmade / "feasibility-and-value.md").is_file(), (
        "前缀相同的手工备份被当成了自动快照 —— 判据被放宽成了 startswith（会静默丢文档）"
    )


def test_the_real_backup_directory_matches_the_expected_shape():
    """真实目录里的候选也必须都长得像自动快照——防正则被放宽成 `doc-*`。"""
    for path in prune_backups.auto_snapshots(BACKUP_ROOT):
        assert prune_backups.AUTO_NAME_RE.match(path.name), f"{path.name} 不该被当成自动快照"


def test_the_real_handmade_snapshot_is_excluded_on_real_data():
    """在**真实仓库**上复核那条反例：前缀同款、名字形状不同 → 必须不在候选里。

    与上面那条造样本的测试是两件事：那条证明"判据能挡住这种形状"，
    这条证明"这个判据真的作用在了真实目录上"。少了后者，
    判据可以是对的、而调用方传的是别的目录（比如扫到了 `backup/` 的父目录）。
    """
    handmade = BACKUP_ROOT / "doc-linenos-mine-230057"
    if not handmade.is_dir():
        pytest.skip("手工快照已被归档到仓库外，无需复核")

    candidates = {p.name for p in prune_backups.auto_snapshots(BACKUP_ROOT)}
    assert handmade.name not in candidates, (
        f"{handmade.name} 里装的是 4 份人工挑的文档，被当成自动快照会被静默清掉"
    )


# --------------------------------------------------------------------------- 2. 保留的是新的
def test_the_newest_snapshots_are_the_ones_kept(tmp_path):
    """保留 = 时间戳最大的 N 份；字典序即可，因为名字里是零填充的定长时间戳。"""
    _make_tree(tmp_path)
    kept, dropped = prune_backups.plan(tmp_path, keep=3)

    assert [p.name for p in kept] == _newest_first(AUTO_NAMES)[:3], "保留的不是最新的三份"
    assert [p.name for p in dropped] == _newest_first(AUTO_NAMES)[3:]
    assert {p.name for p in kept} | {p.name for p in dropped} == set(AUTO_NAMES), "有快照既没留也没清"


def test_nothing_is_dropped_below_the_bound(tmp_path):
    _make_tree(tmp_path, auto=AUTO_NAMES[:2], manual=[])
    kept, dropped = prune_backups.plan(tmp_path, keep=5)

    assert len(kept) == 2 and dropped == []


# --------------------------------------------------------------------------- 3. dry-run 不碰盘
def test_a_dry_run_touches_nothing(tmp_path):
    """默认必须是"只看不动"：否则"先跑一遍看看"就等同于已经清完了。"""
    _make_tree(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())

    kept, dropped, dest = prune_backups.prune(tmp_path, keep=2, apply=False, trash_root=tmp_path / "trash")

    assert len(dropped) == 5 and dest is None
    assert sorted(p.name for p in tmp_path.iterdir()) == before, "dry-run 动了文件系统"
    assert not (tmp_path / "trash").exists(), "dry-run 建了废纸篓目录"


def test_apply_moves_the_dropped_snapshots_into_the_trash(tmp_path):
    """真移是 `mv` 不是 `rm`——清错了要能捞回来。"""
    _make_tree(tmp_path)
    trash = tmp_path / "trash"

    kept, dropped, dest = prune_backups.prune(tmp_path, keep=2, apply=True, trash_root=trash)

    assert dest is not None and dest.is_dir()
    for path in dropped:
        assert not path.exists(), f"{path.name} 还在原处"
        assert (dest / path.name).is_dir(), f"{path.name} 没进废纸篓（被删了？）"
    for path in kept:
        assert path.is_dir(), f"该保留的 {path.name} 被移走了"


# --------------------------------------------------------------------------- 4. 上界判定只有一处
def test_keep_below_one_is_refused_by_the_single_rule():
    """`keep=0` 等于把回退能力一起丢掉，必须是错误而不是"清空"。

    上界判定只允许有一处实现：CLI 的消息必须**就是** `plan()` 抛出来的那条，
    否则以后改一处忘另一处，命令行与实际行为会分叉。
    """
    with pytest.raises(ValueError) as excinfo:
        prune_backups.plan(REPO_ROOT / "artifacts", keep=0)
    assert "--keep" in str(excinfo.value)

    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "prune_backups.py"), "--keep", "0", "--root", str(REPO_ROOT)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, f"退出码应为 2，实际 {proc.returncode}"
    assert proc.stdout.strip() == str(excinfo.value), "CLI 的消息与 plan() 的不一致（判定被抄成了两份）"


def test_cli_lists_without_moving_by_default(tmp_path):
    _make_tree(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "prune_backups.py"), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    assert "待移" in proc.stdout and "dry-run" in proc.stdout
    assert len([p for p in tmp_path.iterdir() if p.is_dir()]) == len(AUTO_NAMES) + len(MANUAL_NAMES)


def _callee(node: ast.Call) -> str:
    """取被调用者的名字，`f()` 与 `shutil.copy2()` 两种写法都要认。

    单看 `ast.Name` 会漏掉所有方法调用——`shutil.copy2` 是 `Attribute`。
    漏掉的表现是"这条断言永远为假"，于是它**看起来在守着备份步骤，实际什么都不守**。
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def test_the_write_path_actually_prunes_after_copying_the_backup():
    """`main()` 的写入路径上必须真的有收敛那一步，且在 `shutil.copy2` **之后**。

    为什么不直接跑 `main()`：它会改写仓库里的真实文档，测试不能干这个。
    所以退一步做**结构性断言**——查 AST。这比断言源码文本稳（注释/换行改动不会误伤），
    又足够钉住"接线还在不在"。

    顺序也是判据的一部分：先备份再收敛，最新那份才会落在保留窗口里；
    反过来则每次都比上界多留一份，日积月累又回到"没有上界"。
    """
    tree = ast.parse((SCRIPTS / "fix_doc_linenos.py").read_text(encoding="utf-8"))
    main_fn = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"), None
    )
    assert main_fn is not None, "fix_doc_linenos.py 里找不到 main()"

    calls = [(n.lineno, _callee(n)) for n in ast.walk(main_fn) if isinstance(n, ast.Call)]
    calls = [(line, name) for line, name in calls if name]
    prune_lines = [line for line, name in calls if name == "_prune_backups"]
    copy_lines = [line for line, name in calls if name == "copy2"]

    assert prune_lines, (
        "main() 里没有调用 _prune_backups —— 回填后不会收敛，"
        "artifacts/backup/ 会继续无上界增长（P2-8 的根因就是这个）"
    )
    assert copy_lines, "main() 里没有 shutil.copy2 —— 备份步骤不见了，回填就不可回退了"
    assert min(copy_lines) < min(prune_lines), (
        f"收敛跑在备份之前（copy2@{copy_lines} vs _prune_backups@{prune_lines}）："
        "最新那份会立刻被算成多余的，每次都比上界多留一份"
    )


# --------------------------------------------------------------------------- 5. 接线：回填后自动收敛
def test_the_backfill_step_calls_the_retention_policy_with_apply(tmp_path, monkeypatch, capsys):
    """`fix_doc_linenos --write` 备份完必须顺手收敛，且**必须传 `apply=True`**。

    漏传 `apply` 的代价真付过一次：接线在、日志在、还打印了「已清掉 1 份」，
    而快照数照旧每次 +1（`prune()` 默认 dry-run）。**机制看着在跑、实际什么都不做**，
    比彻底不接还难查——不接至少不会让人以为已经收敛了。

    这里不真跑回填（那会改写仓库里的文档），只验证备份之后的那个调用点：
    ① 丢给 `_prune_backups` 的是**备份目录的父目录**（不是刚建出来的那一份快照）；
    ② 真的要求落盘执行。
    """
    backup_dir = tmp_path / "artifacts" / "backup" / "doc-linenos-20260929-120000"
    backup_dir.mkdir(parents=True)
    seen: list[tuple[Path, dict]] = []

    def fake_prune(root, **kwargs):
        seen.append((Path(root), kwargs))
        return [], [], None

    monkeypatch.setattr(prune_backups, "prune", fake_prune)
    fix_doc_linenos._prune_backups(backup_dir.parent)

    assert [root for root, _ in seen] == [backup_dir.parent], (
        f"保留策略的扫描面不对：应该扫 {backup_dir.parent}，实际扫了 {seen}。"
        "扫到快照自己内部的话，上界就永远算不对。"
    )
    assert seen[0][1].get("apply") is True, (
        f"没传 apply=True（实际 {seen[0][1]}）—— prune() 默认 dry-run，这一整条接线等于没接"
    )
    assert "保留上界内" in capsys.readouterr().out


def test_the_backfill_step_prunes_for_real(tmp_path, monkeypatch, capsys):
    """端到端钉住"收敛真的发生了"——不是传参对、而是目录真的变小了。

    与上一条的区别：上一条查**参数**，这条查**结果**。参数对但 `prune()` 自己空转
    （比如把 `shutil.move` 写成 `shutil.copy`），只有这条会红。

    把 `Path.home` 指到 `tmp_path`，这样废纸篓落在临时目录里，
    测试**不会往真实的 `~/.Trash` 扔东西**。
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    root = tmp_path / "artifacts" / "backup"
    _make_tree(root, manual=[])

    fix_doc_linenos._prune_backups(root)

    remaining = [p.name for p in root.iterdir()]
    expected = _newest_first(AUTO_NAMES)[: prune_backups.DEFAULT_KEEP]
    assert sorted(remaining, reverse=True) == expected, (
        f"收敛后应只剩最近的 {prune_backups.DEFAULT_KEEP} 份，实际 {sorted(remaining, reverse=True)}"
    )
    trashed = list((tmp_path / ".Trash").glob("prune-backups-*/*"))
    assert len(trashed) == len(AUTO_NAMES) - prune_backups.DEFAULT_KEEP, (
        f"被清掉的 {len(AUTO_NAMES) - prune_backups.DEFAULT_KEEP} 份应进废纸篓，实际 {trashed}"
    )
    assert "已清掉" in capsys.readouterr().out


def test_the_retention_helper_never_claims_to_have_cleaned_when_it_did_not(tmp_path, monkeypatch, capsys):
    """`apply=True` 却没返回废纸篓目录时，**不许**打印"已清掉"。

    这是上一条 bug 的直接护栏：当时它打印了「已清掉 1 份 → None」，
    把一件没做的事说成做了。日志说做过、目录里没变，查起来要从头核对一遍。
    """
    _make_tree(tmp_path, manual=[])
    monkeypatch.setattr(prune_backups, "prune", lambda root, **kw: ([], list(root.iterdir()), None))

    fix_doc_linenos._prune_backups(tmp_path)

    out = capsys.readouterr().out
    assert "已清掉" not in out, f"没真的移动，却说清掉了：{out!r}"
    assert "未被移动" in out, f"没生效也没说清楚：{out!r}"


def test_the_retention_helper_fails_open_and_says_so(tmp_path, monkeypatch, capsys):
    """回填已经写完了，收纳失败不该把回填一起否掉；但**必须**打出来。

    静默 fail-open 在这里格外危险：没清理和清理成功长得一模一样，
    下一次 `--write` 又叠一份，问题会一直藏到磁盘上多出几十份才被看见。
    """
    def boom(_root, **_kwargs):
        raise OSError("Operation not permitted")

    monkeypatch.setattr(prune_backups, "prune", boom)
    fix_doc_linenos._prune_backups(tmp_path)          # 不抛异常 = fail-open

    out = capsys.readouterr().out
    assert "⚠️" in out and "不影响本次回填" in out, f"失败没有打出来：{out!r}"
