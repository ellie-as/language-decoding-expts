#!/usr/bin/env python3
"""Summarize podcast text-window encoding outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent / "outputs" / "text_window_encoding"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_dir)
    parser.add_argument("--min-prefix-channels", type=int, default=4)
    return parser.parse_args()


def window_columns(df: pd.DataFrame) -> list[tuple[int, str]]:
    cols = []
    for col in df.columns:
        if col.startswith("corr_w"):
            cols.append((int(col.replace("corr_w", "")), col))
    return sorted(cols)


def save_overall_summary(df: pd.DataFrame, windows: list[tuple[int, str]], out_dir: Path) -> pd.DataFrame:
    rows = []
    for window, col in windows:
        values = df[col]
        rows.append(
            {
                "window_words": window,
                "mean_r": float(values.mean()),
                "median_r": float(values.median()),
                "q25_r": float(values.quantile(0.25)),
                "q75_r": float(values.quantile(0.75)),
                "max_r": float(values.max()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "overall_window_summary.csv", index=False)
    return out


def save_count_plot(df: pd.DataFrame, windows: list[int], out_dir: Path) -> pd.DataFrame:
    counts = df["preferred_window_words"].value_counts().reindex(windows, fill_value=0).reset_index()
    counts.columns = ["window_words", "n_channels"]
    counts["share_channels"] = counts["n_channels"] / len(df)
    counts.to_csv(out_dir / "preferred_window_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    ax.bar(counts["window_words"].astype(str), counts["n_channels"], color="#4C78A8")
    ax.set_xlabel("preferred trailing window (words)")
    ax.set_ylabel("channels")
    ax.set_title("Preferred text-window horizon")
    fig.savefig(out_dir / "preferred_window_counts.png", dpi=220)
    plt.close(fig)
    return counts


def save_group_heatmap(
    df: pd.DataFrame,
    windows: list[tuple[int, str]],
    group_col: str,
    out_dir: Path,
    min_channels: int,
) -> pd.DataFrame:
    groups = []
    for group, group_df in df.groupby(group_col, sort=True):
        if len(group_df) < min_channels:
            continue
        row = {"group": group, "n_channels": len(group_df)}
        for window, col in windows:
            row[str(window)] = float(group_df[col].mean())
        groups.append(row)
    table = pd.DataFrame(groups)
    if table.empty:
        return table
    table = table.sort_values("n_channels", ascending=False)
    table.to_csv(out_dir / f"{group_col}_window_heatmap_values.csv", index=False)

    mat = table[[str(window) for window, _ in windows]].to_numpy()
    fig_height = max(3.0, 0.32 * len(table) + 1.2)
    fig, ax = plt.subplots(figsize=(7.5, fig_height), constrained_layout=True)
    vmax = float(np.nanmax(np.abs(mat)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(windows)))
    ax.set_xticklabels([str(window) for window, _ in windows])
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels([f"{row.group} (n={int(row.n_channels)})" for row in table.itertuples()])
    ax.set_xlabel("trailing text window (words)")
    ax.set_title(f"Mean encoding r by {group_col}")
    fig.colorbar(im, ax=ax, label="mean r")
    fig.savefig(out_dir / f"{group_col}_window_heatmap.png", dpi=220)
    plt.close(fig)
    return table


def save_lag_scatter(df: pd.DataFrame, out_dir: Path) -> float | None:
    if "gpt2_preferred_lag_s" not in df.columns:
        return None
    valid = df[["gpt2_preferred_lag_s", "preferred_window_words", "best_corr"]].dropna()
    if valid.empty:
        return None
    rho = float(valid["gpt2_preferred_lag_s"].corr(np.log10(valid["preferred_window_words"]), method="spearman"))

    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    sc = ax.scatter(
        valid["gpt2_preferred_lag_s"],
        valid["preferred_window_words"],
        c=valid["best_corr"],
        cmap="inferno_r",
        s=28,
        alpha=0.82,
    )
    ax.set_yscale("log")
    ax.set_xlabel("GPT-2 preferred lag (s)")
    ax.set_ylabel("preferred text window (words)")
    ax.set_title(f"Lag vs context horizon (Spearman rho={rho:.2f})")
    fig.colorbar(sc, ax=ax, label="best r")
    fig.savefig(out_dir / "gpt2_lag_vs_preferred_text_window.png", dpi=220)
    plt.close(fig)
    return rho


def write_markdown(
    *,
    out_dir: Path,
    overall: pd.DataFrame,
    counts: pd.DataFrame,
    family_table: pd.DataFrame,
    prefix_table: pd.DataFrame,
    lag_rho: float | None,
) -> None:
    best_overall = overall.sort_values("mean_r", ascending=False).iloc[0]
    top_count = counts.sort_values("n_channels", ascending=False).iloc[0]
    long_share = counts.loc[counts["window_words"] >= 100, "share_channels"].sum()
    short_share = counts.loc[counts["window_words"] <= 10, "share_channels"].sum()

    lines = [
        "# Text-Window Encoding Summary",
        "",
        f"- Best overall mean r: {int(best_overall.window_words)} words (mean r={best_overall.mean_r:.4f}).",
        f"- Most common preferred window: {int(top_count.window_words)} words ({int(top_count.n_channels)} channels, {top_count.share_channels:.1%}).",
        f"- Short-window preference (<=10 words): {short_share:.1%} of channels.",
        f"- Long-window preference (>=100 words): {long_share:.1%} of channels.",
    ]
    if lag_rho is not None and not np.isnan(lag_rho):
        lines.append(f"- Spearman correlation between GPT-2 preferred lag and preferred text-window size: rho={lag_rho:.3f}.")
    if not family_table.empty:
        lines.extend(["", "## Family Highlights", ""])
        for _, row in family_table.iterrows():
            vals = {int(col): float(row[col]) for col in family_table.columns if str(col).isdigit()}
            best_w = max(vals.items(), key=lambda item: item[1])[0]
            lines.append(f"- {row['group']} (n={int(row['n_channels'])}): strongest mean r at {best_w} words.")
    if not prefix_table.empty:
        top_prefix = prefix_table.head(8)
        lines.extend(["", "## Largest Prefix Groups", ""])
        for _, row in top_prefix.iterrows():
            vals = {int(col): float(row[col]) for col in prefix_table.columns if str(col).isdigit()}
            best_w = max(vals.items(), key=lambda item: item[1])[0]
            lines.append(f"- {row['group']} (n={int(row['n_channels'])}): strongest mean r at {best_w} words.")
    (out_dir / "text_window_interpretation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = args.results_dir
    channel_path = out_dir / "channel_text_window_scores.csv"
    if not channel_path.is_file():
        raise FileNotFoundError(channel_path)
    df = pd.read_csv(channel_path)
    windows = window_columns(df)
    window_values = [window for window, _ in windows]

    overall = save_overall_summary(df, windows, out_dir)
    counts = save_count_plot(df, window_values, out_dir)
    family_table = save_group_heatmap(df, windows, "family", out_dir, min_channels=1)
    prefix_table = save_group_heatmap(df, windows, "prefix", out_dir, min_channels=args.min_prefix_channels)
    lag_rho = save_lag_scatter(df, out_dir)
    write_markdown(
        out_dir=out_dir,
        overall=overall,
        counts=counts,
        family_table=family_table,
        prefix_table=prefix_table,
        lag_rho=lag_rho,
    )
    print(f"Wrote summary outputs to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
