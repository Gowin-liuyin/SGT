"""M8b PII 指标：PII 掩码、字符占比、PII-Ratio 与 PII 片段恢复率。

论文 §7 (p.9) 在 PUPA 上报告

    PII-Ratio = (T* 中 PII 占比) / (T 中 PII 占比)

其中 ``T`` 为原始文本（PUPA 的 ``user_query``，SPEC A20），``T*`` 为攻击重建文本。
论文用 PAPILLON 框架 [49] 的 **LLM-as-judge** 判定 PII（SPEC A10 记录为“需 judge”）。
本模块不调用 judge，改用 PUPA 自带的 ``pii_units``（``||`` 分隔的人工标注 PII 片段）
做**确定性字符级**判定：

- 判定：字符落在任一 ``pii_units`` 片段的一次出现区间内即为 PII 字符；
  匹配大小写不敏感、**最长优先**、互不重叠（左到右扫描，等价于“每个位置先试最长片段”）；
- 占比：PII 字符数 / 文本字符数（``len(text) == 0`` 时为 0.0）；
- PII-Ratio：重建文本的占比 ÷ 原始文本的占比；分母为 0（原始文本无 PII）时**未定义**，
  ``ratio=None``、``defined=False``；
- 片段恢复率（辅助口径）：原始文本中出现的 PII 片段，有多大比例（按片段计）原样
  （大小写不敏感地）出现在重建文本里。

与论文 / PAPILLON judge 的口径差异（必须随结果一起声明，见交付报告）：

1. judge 逐条判定“这段文本里还有哪些 PII”，能识别**改写**、**翻译**（如
   "Rachel Zheng" → 中文译名）与**部分泄露**（如 "John Smith" → "John"）；
   字符匹配只能发现**原样片段**，因此重建文本里的残余 PII 会被低估，PII-Ratio 偏高；
2. PUPA 的 ``pii_units`` 是全局片段表，不是逐位置标注；同一片段在文本里多次出现时，
   本实现按“任一次出现即为 PII 字符”统计（与 ``redacted_query`` 的全量替换一致）；
3. 因此本模块数值**不能**与论文 Table 3 / PAPILLON 的数值直接比较，只能同口径自比。
"""

from __future__ import annotations

import re
from typing import Sequence

__all__ = [
    "split_pii_units",
    "pii_mask",
    "pii_fraction",
    "pii_ratio",
    "pii_units_recovered",
]


# --------------------------------------------------------------------------- #
# 片段表
# --------------------------------------------------------------------------- #
def split_pii_units(pii_units: str | Sequence[str]) -> list[str]:
    """把 PUPA 的 ``pii_units`` 规范成去重片段表。

    ``pii_units`` 在 PUPA 里是 ``"rachel||johnny||h&r technology"`` 这样的字符串，
    按 ``||`` 切分；也接受已切好的序列。切分后逐条 ``strip``、丢弃空串，并按
    **大小写不敏感**去重（保留首次出现的原始写法与顺序）。
    """
    if isinstance(pii_units, str):
        raw: list[str] = pii_units.split("||")
    else:
        raw = []
        for unit in pii_units:
            if not isinstance(unit, str):
                raise TypeError(
                    f"pii_units 的元素必须是 str，实际 {type(unit).__name__}: {unit!r}"
                )
            raw.append(unit)
    units: list[str] = []
    seen: set[str] = set()
    for unit in raw:
        text = unit.strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        units.append(text)
    return units


def _check_text(name: str, value) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} 必须是 str，实际 {type(value).__name__}")
    return value


# --------------------------------------------------------------------------- #
# 掩码与占比
# --------------------------------------------------------------------------- #
def pii_mask(text: str, pii_units: Sequence[str]) -> list[bool]:
    """逐字符标记是否落在某个 PII 片段上（``len(text)`` 个 bool）。

    - 大小写不敏感（``re.IGNORECASE``）；
    - 最长优先：同一位置可匹配多个片段时取最长的（正则分支按长度降序排列，
      等价于逐位置尝试由长到短）；
    - 不重叠：命中后从命中区间之后继续扫描，重复/重叠的片段不会二次计数；
    - 同一片段在文本里出现多次时全部标记。

    例：``pii_mask("New York", ["New York", "York"])`` 返回 8 个 True（不重复计数）。
    """
    text = _check_text("text", text)
    mask = [False] * len(text)
    units = split_pii_units(pii_units)
    if not text or not units:
        return mask
    # 长度降序 → 正则分支在同一位置优先尝试最长片段；长度相同时保持输入顺序。
    ordered = sorted(units, key=len, reverse=True)
    pattern = "|".join(re.escape(unit) for unit in ordered)
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        start, end = match.span()
        for index in range(start, end):
            mask[index] = True
    return mask


def pii_fraction(text: str, pii_units: Sequence[str]) -> float:
    """PII 字符占比：``pii_mask`` 为 True 的字符数 / ``len(text)``；空文本返回 0.0。"""
    text = _check_text("text", text)
    if not text:
        return 0.0
    return sum(pii_mask(text, pii_units)) / len(text)


# --------------------------------------------------------------------------- #
# PII-Ratio 与片段恢复率
# --------------------------------------------------------------------------- #
def pii_ratio(
    original: str, reconstructed: str, pii_units: Sequence[str]
) -> dict:
    """论文定义的 PII-Ratio：重建文本的 PII 占比 ÷ 原始文本的 PII 占比。

    返回 ``{"ratio": float | None, "pii_orig": float, "pii_rec": float, "defined": bool}``：

    - ``pii_orig = 0``（原始文本不含任何 PII 片段）时分母为 0，比值未定义：
      ``ratio=None``、``defined=False``，且按任务单约定 ``pii_orig = pii_rec = 0.0``；
    - 否则 ``ratio = pii_rec / pii_orig``：重建与原文本一样含 PII → 1.0，
      重建把 PII 全脱敏 → 0.0，重建 PII 占比更高 → > 1.0。

    注意比值是**占比之比**：文本长度不同（截断、增删）也会改变比值，因此它刻画的是
    “PII 密度是否下降”，不是“PII 字符数是否下降”。
    """
    original = _check_text("original", original)
    reconstructed = _check_text("reconstructed", reconstructed)
    pii_orig = pii_fraction(original, pii_units)
    if pii_orig <= 0.0:
        return {"ratio": None, "pii_orig": 0.0, "pii_rec": 0.0, "defined": False}
    pii_rec = pii_fraction(reconstructed, pii_units)
    return {
        "ratio": pii_rec / pii_orig,
        "pii_orig": pii_orig,
        "pii_rec": pii_rec,
        "defined": True,
    }


def pii_units_recovered(
    original: str, reconstructed: str, pii_units: Sequence[str]
) -> dict:
    """PII 片段恢复率（辅助口径，按片段计，大小写不敏感）。

    统计原始文本里**出现过**的 PII 片段（去重后）中，有多大比例原样出现在重建文本里。
    返回 ``{"n_units": int, "n_recovered": int, "recovered_frac": float}``；
    ``n_units == 0`` 时 ``recovered_frac = 0.0``。
    """
    original = _check_text("original", original)
    reconstructed = _check_text("reconstructed", reconstructed)
    units = split_pii_units(pii_units)
    folded_original = original.lower()
    folded_reconstructed = reconstructed.lower()
    present = [unit for unit in units if unit.lower() in folded_original]
    recovered = [unit for unit in present if unit.lower() in folded_reconstructed]
    return {
        "n_units": len(present),
        "n_recovered": len(recovered),
        "recovered_frac": (len(recovered) / len(present)) if present else 0.0,
    }
