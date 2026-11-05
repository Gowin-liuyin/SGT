"""S1 安全模型与攻击门禁：威胁模型字段、攻击定义自洽、噪声新鲜性、攻击有效性对照。

读取 `results/attack_*.json` 与 `results/pupa_*_beamclean.json` 作为证据，
另用合成数据现场验证噪声新鲜性与 `fr_by_r` 的定义。
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gates.common import Gate  # noqa: E402

REQUIRED_THREAT = {
    "attacker_position", "attacker_knows", "sgt_weights_hidden",
    "greybox_rank_histogram", "fresh_noise", "whitebox_weights_tested", "attacker_goal",
}


def load_attack(tag: str):
    p = REPO_ROOT / "results" / f"attack_{tag}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    g = Gate("s1", "security", module_entry="§1/§4.1/§6：威胁模型 + NN/MRP/BeamClean + PAC")
    files = sorted(glob.glob(str(REPO_ROOT / "results" / "attack_*.json")))
    g.check("c-files", expected=">=6 个 attack_*.json", actual=len(files), passed=len(files) >= 6)
    if not files:
        return 0 if g.finish() else 1

    data = {Path(f).stem.replace("attack_", ""): json.loads(Path(f).read_text()) for f in files}

    # 1) 威胁模型字段完整且与论文一致
    bad_tm = []
    for tag, d in data.items():
        tm = d.get("threat_model", {})
        missing = REQUIRED_THREAT - set(tm)
        if missing:
            bad_tm.append(f"{tag}:missing={missing}")
            continue
        if not ({"tokenizer", "embedding_table"} <= set(tm["attacker_knows"])):
            bad_tm.append(f"{tag}:knows={tm['attacker_knows']}")
        if tm["sgt_weights_hidden"] is not True or tm["whitebox_weights_tested"] is not False:
            bad_tm.append(f"{tag}:whitebox")
        if tm["greybox_rank_histogram"] is not True:
            bad_tm.append(f"{tag}:greybox")
    g.check("c-threat-model", expected="全部含完整威胁模型（表/tokenizer 已知、SGT 权重未知、灰盒直方图）",
            actual=f"{bad_tm}", passed=not bad_tm)

    # 2) 无保护基线：重建应当完全成功
    base = data.get("baseline")
    ok_base = (base is not None and base["metrics"]["nn_fr"] == 0.0
               and base["metrics"]["mrp_fr"] == 0.0
               and base["metrics"]["ttr_1"] == 0.0
               and base["metrics"]["hist_entropy"] == 0.0
               and base["metrics"]["fr_by_r"]["1"] == 0.0)
    g.check("c-baseline-reconstructable", expected="nn_fr=mrp_fr=ttr_1=hist=0",
            actual=(base["metrics"] if base else None), passed=bool(ok_base))

    # 3) 攻击定义自洽：fr_by_r[r=1] == nn_fr；r=|V| 是"最远邻"攻击
    bad_def = []
    for tag, d in data.items():
        m = d["metrics"]
        if abs(float(m["fr_by_r"]["1"]) - float(m["nn_fr"])) > 1e-9:
            bad_def.append(f"{tag}:fr1={m['fr_by_r']['1']} vs nn={m['nn_fr']}")
    g.check("c-attack-defs", expected="fr_by_r[1] == nn_fr", actual=bad_def, passed=not bad_def)

    # 4) SGT 变体：NN 与 MRP 都被挡住（论文的攻击口径下）
    SGT_MODELS = {"abscos", "cossim", "mi", "mi_abscos", "mi_abscos_norm"}
    sgt_tags = sorted(set(data) & SGT_MODELS)
    weak = [t for t in sgt_tags if data[t]["metrics"]["nn_fr"] < 90.0]
    g.check("c-sgt-resists-nn", expected="5 个 SGT 变体 NN-FR >= 90%", actual=f"weak={weak} n={len(sgt_tags)}",
            passed=bool(sgt_tags) and not weak)

    # 5) 噪声新鲜性：同一输入两次采样不同；同种子调用序列可复现
    from sgt.model import SGTModel
    from sgt.utility import SGTEmbedding
    torch.manual_seed(0)
    base_emb = torch.nn.Embedding(50, 16)
    sgt = SGTModel(16, hidden=8, n_layers=1, n_heads=2, max_len=16, sigma_init=0.5)
    with torch.no_grad():
        for p in sgt.parameters():
            p.normal_(0, 0.1)
    ids = torch.tensor([[1, 2, 3, 4]])
    w1 = SGTEmbedding(base_emb, sgt, seed=0)
    a1, a2 = w1(ids), w1(ids)
    w2 = SGTEmbedding(base_emb, sgt, seed=0)
    b1 = w2(ids)
    g.check("c-fresh-noise", expected="同输入两次不同；同 seed 不同实例首次调用一致",
            actual=f"diff={not torch.equal(a1, a2)} reproducible={torch.equal(a1, b1)}",
            passed=(not torch.equal(a1, a2)) and torch.equal(a1, b1))

    # 6) BeamClean 与 NN 的一致性（论文 Table 3：BC ≈ NN）
    bc_base = REPO_ROOT / "results" / "pupa_baseline_beamclean.json"
    bc_sgt = REPO_ROOT / "results" / "pupa_abscos_beamclean.json"
    ok_bc = bc_base.exists() and bc_sgt.exists()
    detail = {}
    if ok_bc:
        b = json.loads(bc_base.read_text())
        s = json.loads(bc_sgt.read_text())
        detail = {"baseline_nn_fr": b["nn"]["fr"], "baseline_bc_fr": b["beamclean"]["fr"],
                  "sgt_nn_fr": s["nn"]["fr"], "sgt_bc_fr": s["beamclean"]["fr"]}
        ok_bc = (abs(b["beamclean"]["fr"] - b["nn"]["fr"]) <= 5.0
                 and abs(s["beamclean"]["fr"] - s["nn"]["fr"]) <= 5.0
                 and b["beamclean"]["fr"] <= 1.0)
    g.check("c-beamclean-consistency", expected="BC-FR ≈ NN-FR；无保护基线 BC-FR=0", actual=detail,
            passed=ok_bc)

    # 7) PUPA PII：基线 = 1.0，SGT 显著下降
    pii = {}
    for p in sorted((REPO_ROOT / "results").glob("pupa_*.json")):
        d = json.loads(p.read_text())
        pii[p.stem] = d.get("nn", {}).get("pii_ratio_mean")
    ok_pii = pii.get("pupa_baseline") is not None and abs(pii["pupa_baseline"] - 1.0) < 0.05
    g.check("c-pii-baseline", expected="无保护 PII-Ratio ≈ 1.0", actual=pii.get("pupa_baseline"),
            passed=bool(ok_pii))
    sgt_pii = [v for k, v in pii.items() if k not in ("pupa_baseline", "pupa_baseline_beamclean")
               and isinstance(v, (int, float))]
    g.check("c-pii-drop", expected="所有 SGT 变体 PII-Ratio < 0.5", actual=sgt_pii,
            passed=bool(sgt_pii) and all(v < 0.5 for v in sgt_pii))

    ok = g.finish()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
