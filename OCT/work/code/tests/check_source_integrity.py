#!/usr/bin/env python3
"""源码完整性检查：拦截"分块写入导致行被粘连"这一类静默缺陷。

背景
----
本项目代码采用分块方式写入。若某一块结尾缺少换行，下一块的首行会被拼到上一行末尾。
这类缺陷**往往仍是合法语法**，因而能通过 `py_compile`，只在运行期才炸：

* `return out@dataclass` —— 合法（等价于 `out @ dataclass` 矩阵乘），运行期 TypeError；
* `return img, msk@torch.no_grad()` —— 同上，且 `@torch.no_grad()` 装饰器静默失效；
* `return out# ----------` —— 合法（注释），但下一代码块的分隔头被吞掉。

因此在跑任何实验前先做一次静态扫描。四項检查：
  1. 文件以换行结尾（这是避免粘连的前提，也是本次缺陷的根因）；
  2. 无"非空白字符紧跟 @"（装饰器被粘连 / 意外的矩阵乘）；
  3. 无"代码字符紧跟 #"（注释或块头被粘连）；
  4. 无重复的顶层定义（同一段被写入两次的典型症状）。

字符串与注释内容会在检查前被替换为等长空白，因此本文件自身的说明文字与正则字面量
不会造成误报（这也是上一版检查器的问题）。

本脚本同时覆盖 `*.sh` 与 `*.yaml`：它们同样经分块方式写入，其中 shell 脚本的失效模式
更隐蔽——`set -euo pipefail` 若被粘到上一行注释末尾，会静默变成注释的一部分，
`bash -n` 也查不出来，却会让整个脚本失去"出错即停"的保护。

用法
----
    python3 tests/check_source_integrity.py            # 扫描本仓库自有代码
    python3 tests/check_source_integrity.py --root ..  # 指定根目录
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from collections import Counter
from pathlib import Path

# 第三方副本不检查：其中的矩阵乘等写法会大量误报
SKIP_DIRS = {"tta-flow", ".git", "__pycache__", "venv", ".venv", "site-packages"}

GLUED_AT = re.compile(r"\S@")
GLUED_HASH = re.compile(r"[)\w\]\"']#")
# 注意：必须要求"零空白"，否则会把正常的推导式（[x for x in y]）误判为粘连；
# 粘连的本质就是两行被直接拼接，接缝处不会有空白。
GLUED_STMT = re.compile(r"[}\]\)](?:def|class|for|if|elif|else|while|return|import|from)\b")
TOPLEVEL_DEF = re.compile(r"^(?:def|class)\s+([A-Za-z_]\w*)")
# shell 粘连：代码紧跟连续的 #（分隔块头 #### 若与上一行粘连，会变成 xxx#####...）。
# 必须先在剥掉引号与注释后再匹配，否则行首的整行 #### 与 echo "####" 都会误报。
SHELL_GLUED_BLOCK = re.compile(r"\S#{2,}")
SHELL_COMMENT_START = re.compile(r"(?:^|[\s;|&()])#")

_SKIP_TOKENS = tuple(
    t for t in (
        tokenize.STRING,
        tokenize.COMMENT,
        getattr(tokenize, "FSTRING_START", None),
        getattr(tokenize, "FSTRING_MIDDLE", None),
        getattr(tokenize, "FSTRING_END", None),
    ) if t is not None
)


def iter_source_files(root: Path):
    for pat in ("*.py", "*.sh", "*.yaml", "*.yml"):
        for p in sorted(root.rglob(pat)):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            yield p


def blank_literals(text: str) -> list[str]:
    """把字符串/注释内容替换为等长空白，保持行结构不变。"""
    rows = [list(ln) for ln in text.split("\n")]

    def blank(start: tuple[int, int], end: tuple[int, int]) -> None:
        (sr, sc), (er, ec) = start, end
        for r in range(sr, er + 1):
            if not (1 <= r <= len(rows)):
                continue
            row = rows[r - 1]
            lo = sc if r == sr else 0
            hi = ec if r == er else len(row)
            for i in range(lo, min(hi, len(row))):
                row[i] = " "

    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in _SKIP_TOKENS:
                blank(tok.start, tok.end)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # 不完整/损坏的文件也要能给出报告，退回按原始行检查
        return text.split("\n")
    return ["".join(row) for row in rows]


def check_file(path: Path) -> list[str]:
    """返回该文件的问题列表（每条可直接打印）。"""
    problems: list[str] = []
    raw = path.read_text(encoding="utf-8")

    # 0) 语法检查：任何解析失败都说明文件在写入过程中被破坏
    try:
        ast.parse(raw)
    except SyntaxError as e:
        problems.append(f"第 {e.lineno} 行：语法错误 -> {e.msg}")

    if raw and not raw.endswith("\n"):
        problems.append("文件结尾缺少换行 —— 下次追加内容时会与下一行粘连，请补一个换行")

    lines = blank_literals(raw)
    for i, line in enumerate(lines, 1):
        if GLUED_AT.search(line):
            problems.append(f"第 {i} 行：非空白字符紧跟 @（行被粘连） -> {line.strip()[:90]!r}")
        if GLUED_HASH.search(line):
            problems.append(f"第 {i} 行：代码紧跟 #（注释/块头被粘连） -> {line.strip()[:90]!r}")
        if GLUED_STMT.search(line):
            problems.append(f"第 {i} 行：尾部花括号与语句关键字粘连（两行被拼成一行）"
                            f" -> {line.strip()[:90]!r}")

    names = [m.group(1) for m in (TOPLEVEL_DEF.match(ln) for ln in lines) if m]
    for name, cnt in Counter(names).items():
        if cnt > 1:
            problems.append(f"顶层定义 {name!r} 出现 {cnt} 次（疑似重复写入）")

    return problems


def blank_shell_literals(text: str) -> list[str]:
    """剥掉 shell 的引号内容与注释（替换为等长空白），只保留可执行代码骨架。

    这样 `echo "####"`、行首整行 `####`、`${x##*/}` 都不会被误判为粘连，
    而真正的粘连（上一行代码直接接上块头 `内容####...`）**不在词首**，会被保留。
    """
    out: list[str] = []
    for line in text.split("\n"):
        chars = list(line)
        quote: str | None = None
        i = 0
        while i < len(chars):
            c = chars[i]
            if quote is None:
                if c in "\"'":
                    quote = c
                    chars[i] = " "
                elif c == "\\":                 # 反斜杠转义：连同下一个字符一起抹掉
                    chars[i] = " "
                    if i + 1 < len(chars):
                        chars[i + 1] = " "
                        i += 1
            else:
                if c == quote:
                    quote = None
                chars[i] = " "
            i += 1

        stripped = "".join(chars)
        m = SHELL_COMMENT_START.search(stripped)
        if m:                                     # 词首 # 之后全部视为注释
            start = m.end() - 1
            stripped = stripped[:start] + " " * (len(stripped) - start)
        out.append(stripped)
    return out


def check_shell_file(path: Path) -> list[str]:
    """shell 脚本检查：结尾换行、`bash -n` 语法、粘连的分隔块头。"""
    problems: list[str] = []
    raw = path.read_text(encoding="utf-8")

    if raw and not raw.endswith("\n"):
        problems.append("文件结尾缺少换行 —— 下次追加内容时会与下一行粘连，请补一个换行")

    proc = subprocess.run(["bash", "-n", path.as_posix()], capture_output=True, text=True)
    if proc.returncode != 0:
        detail = " / ".join(proc.stderr.strip().splitlines()[:3])
        problems.append(f"bash -n 语法检查失败: {detail}")

    for i, line in enumerate(blank_shell_literals(raw), 1):
        if SHELL_GLUED_BLOCK.search(line):
            problems.append(f"第 {i} 行：代码紧跟连续 #（块头被粘连） -> {line.strip()[:90]!r}")
    return problems


def check_yaml_file(path: Path) -> list[str]:
    """YAML 检查：结尾换行与制表符缩进（YAML 规范禁止用 tab 缩进）。"""
    problems: list[str] = []
    raw = path.read_text(encoding="utf-8")

    if raw and not raw.endswith("\n"):
        problems.append("文件结尾缺少换行 —— 下次追加内容时会与下一行粘连，请补一个换行")

    for i, line in enumerate(raw.split("\n"), 1):
        if line.startswith("\t") or re.match(r"^\s+\t", line):
            problems.append(f"第 {i} 行：缩进使用了制表符（YAML 不允许）")
    return problems


def check_any(path: Path) -> list[str]:
    """按扩展名分派到对应的检查器。"""
    suffix = path.suffix.lower()
    if suffix == ".py":
        return check_file(path)
    if suffix == ".sh":
        return check_shell_file(path)
    return check_yaml_file(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="源码完整性检查")
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parent.parent,
                    help="扫描根目录（默认本仓库 repro/code）")
    ap.add_argument("--quiet", action="store_true", help="只打印有问题的文件")
    args = ap.parse_args()

    files = list(iter_source_files(args.root))
    if not files:
        print(f"❌ 在 {args.root} 未找到任何源文件")
        return 2

    total = 0
    for path in files:
        problems = check_any(path)
        rel = path.relative_to(args.root) if path.is_relative_to(args.root) else path
        if problems:
            total += len(problems)
            print(f"❌ {rel}")
            for p in problems:
                print(f"     - {p}")
        elif not args.quiet:
            print(f"✅ {rel}")

    print()
    if total:
        print(f"❌ 共发现 {total} 个问题，请修复后再跑实验")
        return 1
    print(f"✅ {len(files)} 个文件全部通过完整性检查")
    return 0


if __name__ == "__main__":
    sys.exit(main())
