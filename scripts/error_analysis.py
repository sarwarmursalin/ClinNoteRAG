"""
Error analysis across the three evaluation strategies.
Usage: python scripts/error_analysis.py
Requires: pandas, scipy. No VPN needed.
"""
import pandas as pd
import numpy as np
from scipy.stats import chi2 as chi2_dist


def load_results():
    base = "scripts"
    naive   = pd.read_csv(f"{base}/results_naive_rag.csv")
    norag   = pd.read_csv(f"{base}/results_no_rag.csv")
    agentic = pd.read_csv(f"{base}/results.csv")
    for df in [naive, norag, agentic]:
        df.dropna(subset=["predicted", "ground_truth"], inplace=True)
        for col in ("predicted", "ground_truth", "case_num"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["predicted", "ground_truth", "case_num"], inplace=True)
        df["predicted"]   = df["predicted"].astype(int)
        df["ground_truth"] = df["ground_truth"].astype(int)
        df["case_num"]    = df["case_num"].astype(int)
    return naive, norag, agentic


def case_metrics(df):
    rows = []
    for case_num, g in df.groupby("case_num"):
        tp = ((g.predicted==1)&(g.ground_truth==1)).sum()
        fp = ((g.predicted==1)&(g.ground_truth==0)).sum()
        fn = ((g.predicted==0)&(g.ground_truth==1)).sum()
        tn = ((g.predicted==0)&(g.ground_truth==0)).sum()
        p  = tp/(tp+fp) if (tp+fp) else 0
        r  = tp/(tp+fn) if (tp+fn) else 0
        f1 = 2*p*r/(p+r) if (p+r) else 0
        rows.append({"case": case_num, "tp":tp, "fp":fp, "fn":fn, "tn":tn,
                     "P":round(p,4), "R":round(r,4), "F1":round(f1,4)})
    return pd.DataFrame(rows).sort_values("case")


def overall_f1(df):
    tp = ((df.predicted==1)&(df.ground_truth==1)).sum()
    fp = ((df.predicted==1)&(df.ground_truth==0)).sum()
    fn = ((df.predicted==0)&(df.ground_truth==1)).sum()
    p  = tp/(tp+fp) if (tp+fp) else 0
    r  = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    return round(p,4), round(r,4), round(f1,4)


def mcnemar_test(df_a, df_b, label_a="A", label_b="B"):
    m = df_a[["pn_num","feature_num","predicted","ground_truth"]].rename(
            columns={"predicted": "pred_a"}).merge(
        df_b[["pn_num","feature_num","predicted"]].rename(
            columns={"predicted": "pred_b"}),
        on=["pn_num","feature_num"]
    )
    a_ok = m.pred_a == m.ground_truth
    b_ok = m.pred_b == m.ground_truth
    n01 = ((a_ok) & (~b_ok)).sum()
    n10 = ((~a_ok) & (b_ok)).sum()
    n11 = ((a_ok) & (b_ok)).sum()
    n00 = ((~a_ok) & (~b_ok)).sum()
    stat = (abs(n01 - n10) - 1)**2 / (n01 + n10) if (n01 + n10) else 0
    pval = 1 - chi2_dist.cdf(stat, 1)
    print(f"\nMcNemar's test: {label_a} vs {label_b}  (N={len(m):,})")
    print(f"  {label_a} correct & {label_b} wrong : {n10:6,}")
    print(f"  {label_b} correct & {label_a} wrong : {n01:6,}")
    print(f"  both correct                  : {n11:6,}")
    print(f"  both wrong                    : {n00:6,}")
    print(f"  chi2={stat:.2f}  p={pval:.2e}  -> {'SIGNIFICANT' if pval < 0.05 else 'NOT significant'}")


def hardest_concepts(df, top_n=15):
    grp = df.groupby(["feature_num","concept"]).apply(
        lambda g: pd.Series({
            "total_present": (g.ground_truth==1).sum(),
            "fn": ((g.predicted==0)&(g.ground_truth==1)).sum(),
        })
    ).reset_index()
    grp["fn_rate"] = grp["fn"] / grp["total_present"].clip(lower=1)
    return grp[grp.total_present >= 10].sort_values("fn_rate", ascending=False).head(top_n)


def negation_split(df):
    neg_mask = df.concept.str.lower().str.startswith(
        ("no ", "not ", "lack", "deny", "denies", "negative", "without", "absent")
    )
    return df[neg_mask], df[~neg_mask]


if __name__ == "__main__":
    naive, norag, agentic = load_results()

    print("=== OVERALL F1 ===")
    for name, df in [("No-RAG", norag), ("Naive RAG", naive), ("Agentic RAG", agentic)]:
        p, r, f1 = overall_f1(df)
        print(f"{name:15}: P={p}  R={r}  F1={f1}")

    print("\n=== PER-CASE F1 COMPARISON ===")
    cm_naive   = case_metrics(naive)
    cm_norag   = case_metrics(norag)
    cm_agentic = case_metrics(agentic)
    compare = (
        cm_norag[["case","F1"]].rename(columns={"F1":"no_rag"})
        .merge(cm_naive[["case","F1"]].rename(columns={"F1":"naive_rag"}), on="case")
        .merge(cm_agentic[["case","F1"]].rename(columns={"F1":"agentic_rag"}), on="case", how="left")
    )
    compare["best"] = compare[["no_rag","naive_rag","agentic_rag"]].idxmax(axis=1)
    print(compare.to_string(index=False))

    mcnemar_test(norag, naive, "no_rag", "naive_rag")

    print("\n=== TOP 15 HARDEST CONCEPTS (no_rag, by FN rate) ===")
    print(hardest_concepts(norag).to_string(index=False))

    print("\n=== NEGATION vs POSITIVE CONCEPT F1 (no_rag) ===")
    neg, pos = negation_split(norag)
    for label, df in [("Negation", neg), ("Positive", pos)]:
        p, r, f1 = overall_f1(df)
        print(f"{label:12} (n={len(df):,}): P={p}  R={r}  F1={f1}")
