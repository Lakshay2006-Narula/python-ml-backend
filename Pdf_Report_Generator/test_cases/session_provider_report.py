"""Per-session provider report for a project.

Produces CSV `data/processed/session_provider_report.csv` with per-session provider counts
and `data/processed/provider_variants.csv` with overall provider variant mapping.

Run:
    $env:PYTHONPATH='.'; python -m test_cases.session_provider_report
"""
from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict
from src.load_data_db import load_project_data


OUT_DIR = Path("data") / "processed"
OUT_CSV = OUT_DIR / "session_provider_report.csv"
OUT_VARIANTS = OUT_DIR / "provider_variants.csv"


def normalize_provider(s: str) -> str:
    if s is None:
        return "<na>"
    s = str(s).strip()
    if s == "":
        return "<na>"
    # simple normalization: lower-case, remove extra whitespace
    return " ".join(s.lower().split())


def build_report(project_id: int = 149):
    raw_df, filtered_df, project = load_project_data(project_id)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if raw_df is None or raw_df.empty:
        print("No raw data for project", project_id)
        return

    # prepare provider column normalized
    df = raw_df.copy()
    df["m_alpha_long_norm"] = df["m_alpha_long"].apply(normalize_provider)

    session_rows = []
    overall_counter = Counter()
    variant_map = defaultdict(Counter)

    for sid, g in df.groupby("session_id"):
        prov_counts = g["m_alpha_long_norm"].value_counts()
        total = prov_counts.sum()
        for prov, cnt in prov_counts.items():
            session_rows.append({
                "session_id": int(sid),
                "provider": prov,
                "count": int(cnt),
                "percent": float(cnt) / float(total) if total else 0.0,
                "total_samples": int(total),
            })
            overall_counter[prov] += int(cnt)
            variant_map[prov][prov] += int(cnt)

    # Build provider variants by grouping original raw strings to normalized form
    # (also capture raw variants)
    raw_variants = raw_df[["m_alpha_long"]].fillna("<na>")
    raw_variants["norm"] = raw_variants["m_alpha_long"].apply(normalize_provider)
    variants = (
        raw_variants.groupby("norm")["m_alpha_long"].value_counts().rename("count").reset_index()
    )
    variants = variants.rename(columns={"m_alpha_long": "raw_value", "norm": "normalized"})

    # Save session-level report
    df_report = pd.DataFrame(session_rows)
    df_report = df_report.sort_values(["session_id", "percent"], ascending=[True, False])
    df_report.to_csv(OUT_CSV, index=False)

    # Save variants
    variants.to_csv(OUT_VARIANTS, index=False)

    # Print concise summary
    print(f"Project {project_id} provider summary saved to: {OUT_CSV}")
    print(f"Provider variant mapping saved to: {OUT_VARIANTS}")

    # Top providers overall
    print('\nTop providers overall:')
    for prov, cnt in overall_counter.most_common():
        print(f"  {prov}: {cnt} samples")

    # Print per-session dominant provider
    print('\nPer-session dominant provider:')
    dom = df_report[df_report.groupby('session_id')['percent'].transform('max') == df_report['percent']]
    for _, r in dom.sort_values('session_id').iterrows():
        print(f"  Session {int(r['session_id'])}: {r['provider']} ({int(r['count'])}/{int(r['total_samples'])} samples, {r['percent']:.2%})")


if __name__ == '__main__':
    build_report()
