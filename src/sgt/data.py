"""M5 数据管线：OpenOrca / Alpaca / PUPA 加载、prompt 构造、tokenize 与 batching。

数据获取约定（任务单 M5）：

- OpenOrca / Alpaca 只走 Hugging Face datasets-server 的逐页 rows 接口
  （``GET /rows?dataset=...&offset=...&length=100``），**不下载 parquet**
  （本机带宽约 1 MB/s，1 GB+ 的 parquet 不可接受）。
- 逐页请求之间至少间隔 ``PAGE_INTERVAL_SECONDS`` 秒；若仍遇 429，则按
  ``RATE_LIMIT_DELAYS`` 长退避（CDN 限流窗口实测约 1–3 分钟）。
- PUPA 文件很小（< 1 MB），优先 ``datasets.load_dataset``；失败时回退到直接
  下载 CSV（``https://huggingface.co/datasets/Columbia-NLP/PUPA/resolve/main/...``）。
  PUPA 文本字段按 SPEC A20 固定为 ``user_query``（含 PII 的原始用户输入）。
- 返回顺序由数据集原始顺序决定，所有函数确定性；OpenOrca / Alpaca 缓存写入
  ``data_cache/<dataset>/train_first{N}.jsonl``（每行一个样本），并记录
  ``meta.json``（n_total / n_fetched / 抓取时间 / 每页 offset）。

论文协议与降规模（SPEC D2/A13）：OpenOrca 的 15% 只作为协议记录，
由 :func:`openorca_counts` 返回 ``n_total`` 与 ``n_selected = ceil(0.15*n_total)``；
实际加载用 ``max_examples`` 截断，``max_examples=None`` 时采用与
``configs/data.yaml`` 一致的降规模默认（OpenOrca 4000 / Alpaca 1000），
不是论文全量。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = _REPO_ROOT / "data_cache"

ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100  # datasets-server rows 接口单页上限
MAX_RETRIES = 3  # 失败重试 3 次（普通错误退避 2/4/8 s）
RETRY_BASE_DELAY = 2.0
REQUEST_TIMEOUT = 30.0
# 逐页请求之间的最小间隔：datasets-server 前置 CloudFront 对突发请求返回 429
# （实测：24 页 ≈40 s、33 页 ≈160 s 均触发，恢复窗口约 1–3 分钟）。
PAGE_INTERVAL_SECONDS = 6.0
# 429 专用退避：CDN 限流窗口长于普通网络抖动，沿用 3 次重试但等待更久。
RATE_LIMIT_DELAYS = (30.0, 60.0, 120.0)

OPENORCA_DATASET = "Open-Orca/OpenOrca"
OPENORCA_CONFIG = "default"
OPENORCA_SPLIT = "train"
OPENORCA_FIELDS = ("id", "system_prompt", "question", "response")
DEFAULT_OPENORCA_MAX_EXAMPLES = 4000
DEFAULT_FRAC = 0.15

ALPACA_DATASET = "tatsu-lab/alpaca"
ALPACA_CONFIG = "default"
ALPACA_SPLIT = "train"
ALPACA_FIELDS = ("instruction", "input", "output", "text")
DEFAULT_ALPACA_MAX_EXAMPLES = 1000

PUPA_DATASET = "Columbia-NLP/PUPA"
PUPA_CONFIGS = {"pupa_tnb": "PUPA_TNB.csv", "pupa_new": "PUPA_New.csv"}
# SPEC A20：PUPA 文本字段固定为 user_query（含 PII 的原始用户输入）。
# 实测字段均值长度（pupa_tnb / pupa_new，字符）：user_query 1201.6 / 1221.5，
# target_response 1329.9 / 1638.7，redacted_query 1049.6 / 1191.7。
# 未按“最长字段”选 target_response 的理由见 SPEC A20：SGT 保护的是输入，
# redacted_query 已脱敏（PII-Ratio 失去意义），target_response 是模型输出。
PUPA_TEXT_FIELD = "user_query"
PUPA_CSV_URL = "https://huggingface.co/datasets/Columbia-NLP/PUPA/resolve/main/{name}"

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler())
_OPENER.addheaders = [("User-Agent", "sgt-repro-m5/1.0")]

_DATASET_ALIASES = {
    "openorca": "openorca",
    "open-orca": "openorca",
    "open_orca": "openorca",
    "open-orca/openorca": "openorca",
    "openorca/openorca": "openorca",
    "alpaca": "alpaca",
    "tatsu-lab/alpaca": "alpaca",
    "pupa": "pupa",
    "pupa_tnb": "pupa",
    "pupa_new": "pupa",
    "columbia-nlp/pupa": "pupa",
}


class ExampleList(list):
    """``list[dict]`` 子类，附带 :attr:`meta`（n_total / n_fetched / 抓取来源等）。"""

    def __init__(self, iterable: Sequence[dict] = (), *, meta: dict | None = None):
        super().__init__(iterable)
        self.meta: dict = dict(meta or {})


# --------------------------------------------------------------------------- #
# HTTP（带重试与指数退避）
# --------------------------------------------------------------------------- #


def _fetch_bytes(url: str) -> bytes:
    """GET ``url``，失败重试 ``MAX_RETRIES`` 次，超时 30 s。

    普通网络错误/5xx 退避 2/4/8 s（任务单约定）；429 是 CDN 限流，按
    ``RATE_LIMIT_DELAYS``（30/60/120 s，实测恢复窗口 1–3 分钟）退避。
    """
    last: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            with _OPENER.open(url, timeout=REQUEST_TIMEOUT) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            # 4xx（除 429）不是瞬时错误，直接失败；5xx/429 重试。
            if exc.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {url}") from exc
            last = exc
            delay = (
                RATE_LIMIT_DELAYS[min(attempt, len(RATE_LIMIT_DELAYS) - 1)]
                if exc.code == 429
                else RETRY_BASE_DELAY * (2**attempt)
            )
            if attempt < MAX_RETRIES:
                time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BASE_DELAY * (2**attempt))
    raise RuntimeError(f"请求失败（{MAX_RETRIES} 次重试后）: {url}: {last!r}") from last


def _fetch_json(url: str) -> Any:
    return json.loads(_fetch_bytes(url).decode("utf-8"))


def _rows_url(dataset: str, config: str, split: str, offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": int(offset),
            "length": int(length),
        }
    )
    return f"{ROWS_ENDPOINT}?{query}"


def _rows_page(dataset: str, config: str, split: str, offset: int, length: int) -> dict:
    """取 datasets-server rows 接口的一页，返回原始 JSON（含 rows/num_rows_total）。"""
    if not 1 <= length <= PAGE_SIZE:
        raise ValueError(f"length 必须在 [1, {PAGE_SIZE}]，实际 {length}")
    payload = _fetch_json(_rows_url(dataset, config, split, offset, length))
    if not isinstance(payload, dict) or "rows" not in payload:
        raise RuntimeError(f"rows 接口响应缺少 rows 字段: {payload!r:.300}")
    if "num_rows_total" not in payload:
        raise RuntimeError(f"rows 接口响应缺少 num_rows_total: {payload!r:.300}")
    return payload


# --------------------------------------------------------------------------- #
# 缓存工具
# --------------------------------------------------------------------------- #


def _resolve_cache_dir(cache_dir: str | Path | None) -> Path:
    if cache_dir is None:
        return DEFAULT_CACHE_DIR
    return Path(cache_dir).expanduser().resolve()


def _read_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except (OSError, ValueError):
        return None


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _read_jsonl(path: Path, n: int) -> list[dict] | None:
    """读取缓存 JSONL；条数不符或解析失败返回 None（触发重取）。"""
    if not path.exists():
        return None
    records: list[dict] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    return None
                records.append(obj)
    except (OSError, ValueError):
        return None
    if len(records) != n:
        return None
    return records


def _write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _cached_jsonl_count(base: Path) -> int:
    """缓存目录下 ``train_first*.jsonl`` 的最大条数（0 表示无缓存）。"""
    best = 0
    for path in base.glob("train_first*.jsonl"):
        try:
            with path.open(encoding="utf-8") as f:
                count = sum(1 for line in f if line.strip())
        except OSError:
            continue
        best = max(best, count)
    return best


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# rows 接口分页加载（OpenOrca / Alpaca 共用）
# --------------------------------------------------------------------------- #


def _load_rows_dataset(
    *,
    dataset: str,
    config: str,
    split: str,
    fields: Sequence[str],
    requested: int,
    cache_dir: Path,
    subdir: str,
    force_refresh: bool = False,
    extra_meta: dict | None = None,
) -> ExampleList:
    if requested <= 0:
        raise ValueError(f"max_examples 必须为正整数，实际 {requested}")
    base = cache_dir / subdir
    base.mkdir(parents=True, exist_ok=True)
    meta_path = base / "meta.json"

    cached_meta = _read_json(meta_path) or {}
    n_total = int(cached_meta["n_total"]) if cached_meta.get("n_total") else None

    # 缓存命中：meta.json 提供了 n_total，直接按条数检查对应缓存文件。
    if not force_refresh and n_total is not None:
        n = min(requested, n_total)
        path = base / f"train_first{n}.jsonl"
        records = _read_jsonl(path, n)
        if records is not None:
            meta = dict(cached_meta)
            meta.update(extra_meta or {})
            meta.update(
                {
                    "n_total": n_total,
                    "n_fetched": n,
                    "requested": requested,
                    "cache_file": str(path),
                }
            )
            meta.setdefault("pages", [])
            meta.setdefault("source", "datasets-server.huggingface.co/rows")
            return ExampleList(records, meta=meta)

    # 取第一页：既用于获得 num_rows_total，也是数据页。
    first = _rows_page(dataset, config, split, 0, min(PAGE_SIZE, requested))
    n_total = int(first["num_rows_total"])
    n = min(requested, n_total)
    path = base / f"train_first{n}.jsonl"
    if not force_refresh:
        records = _read_jsonl(path, n)
        if records is not None:
            meta = _build_meta(
                dataset, config, split, n_total, n, requested, [], path, extra_meta
            )
            return ExampleList(records, meta=meta)

    records: list[dict] = []
    offsets: list[int] = []
    page = first
    offset = 0
    while len(records) < n:
        rows = page.get("rows") or []
        if not rows:
            break
        offsets.append(offset)
        for item in rows:
            row = item["row"]
            records.append({field: row.get(field) for field in fields})
            if len(records) >= n:
                break
        offset += len(rows)
        if len(records) >= n:
            break
        time.sleep(PAGE_INTERVAL_SECONDS)  # 避免 CloudFront 突发限流（429）
        page = _rows_page(
            dataset, config, split, offset, min(PAGE_SIZE, n - len(records))
        )

    if len(records) != n:
        raise RuntimeError(
            f"{dataset}: 期望 {n} 条（n_total={n_total}），实际只取到 {len(records)} 条"
        )

    _write_jsonl(path, records)
    meta = _build_meta(
        dataset, config, split, n_total, len(records), requested, offsets, path, extra_meta
    )
    _write_json(meta_path, meta)
    return ExampleList(records, meta=meta)


def _build_meta(
    dataset: str,
    config: str,
    split: str,
    n_total: int,
    n_fetched: int,
    requested: int,
    pages: Sequence[int],
    cache_path: Path,
    extra_meta: dict | None,
) -> dict:
    meta = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "n_total": int(n_total),
        "n_fetched": int(n_fetched),
        "requested": int(requested),
        "pages": list(pages),
        "fetched_at": _now_iso(),
        "source": "datasets-server.huggingface.co/rows",
        "cache_file": str(cache_path),
    }
    if extra_meta:
        meta.update(extra_meta)
    return meta


# --------------------------------------------------------------------------- #
# OpenOrca
# --------------------------------------------------------------------------- #


def openorca_counts(frac: float = DEFAULT_FRAC) -> dict:
    """论文协议计数：OpenOrca 总量、15% 选中条数、当前缓存条数。

    ``n_selected`` 仅按论文 §7（15% 的 OpenOrca）计算并记录；实际加载由
    :func:`load_openorca` 的 ``max_examples`` 截断（SPEC D2 降规模）。
    """
    if not 0.0 < float(frac) <= 1.0:
        raise ValueError(f"frac 必须在 (0, 1]，实际 {frac}")
    page = _rows_page(OPENORCA_DATASET, OPENORCA_CONFIG, OPENORCA_SPLIT, 0, 1)
    n_total = int(page["num_rows_total"])
    return {
        "n_total": n_total,
        "n_selected": math.ceil(float(frac) * n_total),
        "frac": float(frac),
        "n_fetched": _cached_jsonl_count(DEFAULT_CACHE_DIR / "openorca"),
    }


def load_openorca(
    frac: float = DEFAULT_FRAC,
    *,
    max_examples: int | None = None,
    cache_dir: str | Path | None = None,
    force_refresh: bool = False,
) -> ExampleList:
    """加载 OpenOrca 前 ``max_examples`` 条（原始顺序），并缓存。

    - ``frac`` 只写入 :attr:`ExampleList.meta`（论文协议 15%），不影响选取条数。
    - ``max_examples=None`` 时用降规模默认 ``DEFAULT_OPENORCA_MAX_EXAMPLES=4000``
      （与 ``configs/data.yaml`` 一致）；条数上限还受 ``num_rows_total`` 约束。
    - 返回 ``list[dict]``（``ExampleList``），字段 ``id/system_prompt/question/response``；
      ``.meta`` 含 n_total / n_fetched / pages / n_selected 等。
    """
    if not 0.0 < float(frac) <= 1.0:
        raise ValueError(f"frac 必须在 (0, 1]，实际 {frac}")
    requested = (
        DEFAULT_OPENORCA_MAX_EXAMPLES if max_examples is None else int(max_examples)
    )
    cache = _resolve_cache_dir(cache_dir)
    examples = _load_rows_dataset(
        dataset=OPENORCA_DATASET,
        config=OPENORCA_CONFIG,
        split=OPENORCA_SPLIT,
        fields=OPENORCA_FIELDS,
        requested=requested,
        cache_dir=cache,
        subdir="openorca",
        force_refresh=force_refresh,
        extra_meta={"frac": float(frac)},
    )
    examples.meta["frac"] = float(frac)
    examples.meta["n_selected"] = math.ceil(float(frac) * examples.meta["n_total"])
    return examples


# --------------------------------------------------------------------------- #
# Alpaca
# --------------------------------------------------------------------------- #


def load_alpaca(
    *,
    max_examples: int | None = None,
    cache_dir: str | Path | None = None,
) -> ExampleList:
    """加载 Alpaca 前 ``max_examples`` 条（原始顺序），并缓存。

    ``max_examples=None`` 时用降规模默认 ``DEFAULT_ALPACA_MAX_EXAMPLES=1000``
    （与 ``configs/data.yaml`` 一致；全量为 52002 条）。
    字段 ``instruction/input/output/text``。
    """
    requested = (
        DEFAULT_ALPACA_MAX_EXAMPLES if max_examples is None else int(max_examples)
    )
    cache = _resolve_cache_dir(cache_dir)
    return _load_rows_dataset(
        dataset=ALPACA_DATASET,
        config=ALPACA_CONFIG,
        split=ALPACA_SPLIT,
        fields=ALPACA_FIELDS,
        requested=requested,
        cache_dir=cache,
        subdir="alpaca",
    )


# --------------------------------------------------------------------------- #
# PUPA
# --------------------------------------------------------------------------- #


def _load_pupa_csv(config: str, cache_dir: Path | None = None) -> list[dict]:
    """回退路径：直接下载 PUPA CSV（< 5 MB）并用 csv 解析。"""
    filename = PUPA_CONFIGS[config]
    base = (cache_dir or DEFAULT_CACHE_DIR) / "pupa"
    base.mkdir(parents=True, exist_ok=True)
    path = base / filename
    if not path.exists():
        data = _fetch_bytes(PUPA_CSV_URL.format(name=filename))
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def load_pupa(
    *, config: str = "pupa_tnb", max_examples: int | None = None
) -> ExampleList:
    """加载 PUPA（``pupa_tnb`` 237 条 / ``pupa_new`` 664 条），返回 ``list[dict]``。

    优先 ``datasets.load_dataset("Columbia-NLP/PUPA", config, split="train")``；
    失败时回退到直接下载 CSV。字段：
    ``conversation_hash, predicted_category, user_query, target_response,
    pii_units, redacted_query``；文本字段按 SPEC A20 固定为 ``user_query``。
    """
    if config not in PUPA_CONFIGS:
        raise ValueError(
            f"未知 PUPA config {config!r}，可选 {sorted(PUPA_CONFIGS)}"
        )
    source = "datasets.load_dataset"
    try:
        from datasets import load_dataset

        dataset = load_dataset(PUPA_DATASET, config, split="train")
        records = [dict(row) for row in dataset]
    except Exception:  # noqa: BLE001 - 任何加载失败都回退到 CSV
        records = _load_pupa_csv(config)
        source = "csv"

    n_total = len(records)
    if max_examples is not None:
        if int(max_examples) <= 0:
            raise ValueError(f"max_examples 必须为正整数，实际 {max_examples}")
        records = records[: int(max_examples)]
    return ExampleList(
        records,
        meta={
            "dataset": PUPA_DATASET,
            "config": config,
            "split": "train",
            "n_total": n_total,
            "n_fetched": len(records),
            "source": source,
            "text_field": PUPA_TEXT_FIELD,
        },
    )


# --------------------------------------------------------------------------- #
# prompt / tokenize / batching
# --------------------------------------------------------------------------- #


def _normalise_dataset(dataset: str) -> str:
    if not isinstance(dataset, str):
        raise TypeError(f"dataset 必须是字符串，实际 {type(dataset).__name__}")
    key = dataset.strip().lower()
    if key not in _DATASET_ALIASES:
        raise ValueError(f"未知数据集 {dataset!r}，可选 openorca/alpaca/pupa")
    return _DATASET_ALIASES[key]


def _infer_dataset(example: dict) -> str:
    keys = set(example)
    if {"system_prompt", "question", "response"} <= keys:
        return "openorca"
    if {"instruction", "output"} <= keys:
        return "alpaca"
    if {"user_query", "target_response", "redacted_query"} & keys:
        return "pupa"
    raise ValueError(f"无法从字段推断数据集: {sorted(keys)}")


def build_prompt(example: dict, dataset: str) -> str:
    """按数据集格式拼接 prompt。

    - OpenOrca: ``system_prompt + "\\n" + question + "\\n" + response``（空字段跳过）
    - Alpaca:   ``instruction + ("\\n" + input if input else "") + "\\n" + output``
    - PUPA:     数据集自带文本字段 ``user_query``（SPEC A20：含 PII 的原始输入）；
      缺失或为空时抛 ``ValueError``，不回退到其他字段。
    """
    name = _normalise_dataset(dataset)
    example = example or {}
    if name == "openorca":
        parts = [
            example.get("system_prompt"),
            example.get("question"),
            example.get("response"),
        ]
        return "\n".join(str(part) for part in parts if part)
    if name == "alpaca":
        instruction = example.get("instruction") or ""
        inp = example.get("input") or ""
        output = example.get("output") or ""
        prompt = instruction
        if inp:
            prompt = f"{prompt}\n{inp}"
        return f"{prompt}\n{output}"
    # pupa（SPEC A20：固定 user_query）
    value = example.get(PUPA_TEXT_FIELD)
    if value is None or str(value).strip() == "":
        raise ValueError(f"PUPA 样例缺少非空文本字段 {PUPA_TEXT_FIELD!r}")
    return str(value)


def _to_texts(examples: Sequence[str | dict], dataset: str | None) -> list[str]:
    if isinstance(examples, (str, dict)):
        raise TypeError("examples 必须是序列（list/tuple），不是单个样本")
    texts: list[str] = []
    for example in examples:
        if isinstance(example, str):
            texts.append(example)
        elif isinstance(example, dict):
            name = dataset if dataset is not None else _infer_dataset(example)
            texts.append(build_prompt(example, name))
        else:
            raise TypeError(
                f"样例必须是 str 或 dict，实际 {type(example).__name__}"
            )
    return texts


def tokenize_examples(
    examples: Sequence[str | dict],
    tokenizer,
    *,
    max_len: int = 128,
    dataset: str | None = None,
) -> dict:
    """右截断地 tokenize，返回 ``{"input_ids": [[...]], "attention_mask": [[...]]}``。

    ``examples`` 可以是 prompt 字符串，也可以是数据集样本 dict（此时用
    :func:`build_prompt` 拼接，``dataset`` 缺省时按字段自动推断）。
    """
    if int(max_len) <= 0:
        raise ValueError(f"max_len 必须为正整数，实际 {max_len}")
    texts = _to_texts(examples, dataset)
    if not texts:
        return {"input_ids": [], "attention_mask": []}
    encoded = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
    input_ids = [list(ids)[: int(max_len)] for ids in encoded["input_ids"]]
    attention_mask = [
        list(mask)[: int(max_len)] for mask in encoded["attention_mask"]
    ]
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def tokenize_batch(
    examples: Sequence[str | dict],
    tokenizer,
    *,
    max_len: int = 128,
    dataset: str | None = None,
) -> dict:
    """SPEC §6 命名的别名，等价于 :func:`tokenize_examples`。"""
    return tokenize_examples(
        examples, tokenizer, max_len=max_len, dataset=dataset
    )


def _pad_token_id(tokenizer) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    if pad is None:
        pad = 0
    return int(pad)


def make_batches(
    examples: Sequence[str | dict],
    tokenizer,
    *,
    batch_size: int,
    max_len: int,
    seed: int = 0,
    dataset_name: str | None = None,
    shuffle: bool = True,
) -> Iterator[dict]:
    """按 ``batch_size`` 产出右 padding 的 batch（同 seed 顺序一致）。

    每个 batch 为 ``{"input_ids": LongTensor[B,T],
    "attention_mask": LongTensor[B,T], "texts": list[str]}``，``T`` 为该
    batch 内最长序列（≤ ``max_len``），不足处用 tokenizer 的 ``pad_token_id``
    右填充且 mask=0。``shuffle`` 使用 ``random.Random(seed)`` 的确定性洗牌。
    """
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size 必须为正整数，实际 {batch_size}")
    if int(max_len) <= 0:
        raise ValueError(f"max_len 必须为正整数，实际 {max_len}")
    if int(seed) < 0:
        raise ValueError(f"seed 必须非负，实际 {seed}")

    texts = _to_texts(examples, dataset_name)
    order = list(range(len(texts)))
    if shuffle:
        random.Random(int(seed)).shuffle(order)
    ordered_texts = [texts[i] for i in order]
    encoded = tokenize_examples(ordered_texts, tokenizer, max_len=max_len)
    pad_id = _pad_token_id(tokenizer)

    for start in range(0, len(ordered_texts), int(batch_size)):
        stop = start + int(batch_size)
        batch_ids = encoded["input_ids"][start:stop]
        batch_mask = encoded["attention_mask"][start:stop]
        if not batch_ids:  # pragma: no cover - range 保证非空
            continue
        width = max(len(ids) for ids in batch_ids)
        padded_ids = [ids + [pad_id] * (width - len(ids)) for ids in batch_ids]
        padded_mask = [mask + [0] * (width - len(mask)) for mask in batch_mask]
        yield {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            "texts": ordered_texts[start:stop],
        }


def dataset_fingerprint(examples: Sequence[str | dict], *, n: int = 1000) -> str:
    """前 ``n`` 条 prompt 拼接后的 sha256 前 16 位 hex（用于确定性/缓存校验）。"""
    if int(n) < 0:
        raise ValueError(f"n 必须非负，实际 {n}")
    texts = _to_texts(list(examples)[: int(n)], None)
    return hashlib.sha256("".join(texts).encode("utf-8")).hexdigest()[:16]
