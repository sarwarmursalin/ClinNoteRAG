"""Generate pipeline diagrams for all three RAG strategies + codebase map."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

OUT_DIR = Path(__file__).parent

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "input":    ("#D6EAF8", "#2980B9"),   # light blue / border
    "llm":      ("#D5F5E3", "#1E8449"),   # light green / border
    "db":       ("#FDEBD0", "#E67E22"),   # light orange / border
    "output":   ("#E8DAEF", "#7D3C98"),   # light purple / border
    "agent":    ("#FADBD8", "#C0392B"),   # light red / border
    "decision": ("#FEF9E7", "#B7950B"),   # light yellow / border
    "arrow":    "#5D6D7E",
    "bg":       "#F8F9FA",
}


def box(ax, x, y, w, h, label, sublabel=None,
        facecolor="#D6EAF8", edgecolor="#2980B9",
        fontsize=9, bold=False):
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.6, zorder=3
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    if sublabel:
        ax.text(x, y + 0.04, label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color="#1a252f", zorder=4)
        ax.text(x, y - 0.10, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color="#555", zorder=4, style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color="#1a252f", zorder=4,
                multialignment="center")


def arrow(ax, x1, y1, x2, y2, label=None, color="#5D6D7E"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.4, mutation_scale=14), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.04, my, label, fontsize=7.5, color=color,
                ha="left", va="center", style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Three-strategy comparison (side by side)
# ─────────────────────────────────────────────────────────────────────────────

def make_pipeline_comparison():
    """Clean, readable pipeline comparison — minimal text inside boxes."""

    fig, axes = plt.subplots(1, 3, figsize=(21, 13))
    fig.patch.set_facecolor("#FFFFFF")
    fig.suptitle("ClinNoteRAG — Pipeline Strategy Comparison",
                 fontsize=18, fontweight="bold", color="#111827", y=0.98)

    STRAT = [
        ("No-RAG",      "#059669", "#D1FAE5", "#6EE7B7"),
        ("Naive RAG",   "#2563EB", "#DBEAFE", "#93C5FD"),
        ("Agentic RAG", "#DC2626", "#FEE2E2", "#FCA5A5"),
    ]

    # ── Helper: clean rounded box with ONE short label ─────────────────────────
    def node(ax, cx, cy, w, h, label, fc, ec, fs=11, bold=True):
        rect = mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.04",
            facecolor=fc, edgecolor=ec,
            linewidth=2.0, zorder=3
        )
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color="#111827", zorder=4, multialignment="center")

    def ann(ax, cy, text, color="#6B7280"):
        """Right-side annotation next to a node."""
        ax.text(0.88, cy, text, ha="left", va="center",
                fontsize=8.5, color=color, style="italic", zorder=4)

    def arr(ax, x1, y1, x2, y2, color="#6B7280", lw=2.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=lw, mutation_scale=16), zorder=2)

    def stat_footer(ax, lines, color):
        for i, line in enumerate(lines):
            ax.text(0.5, 0.055 - i * 0.038, line,
                    ha="center", va="center", fontsize=9.5,
                    color=color, fontweight="bold" if i == 0 else "normal")

    NW, NH = 0.74, 0.095   # node width / height

    for ax, (title, tc, fc_main, ec_main) in zip(axes, STRAT):
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor("#FFFFFF")

        # Coloured header bar
        header = mpatches.FancyBboxPatch(
            (0.04, 0.915), 0.97, 0.065,
            boxstyle="round,pad=0.01",
            facecolor=fc_main, edgecolor=ec_main, linewidth=1.5, zorder=3
        )
        ax.add_patch(header)
        ax.text(0.525, 0.948, title,
                ha="center", va="center",
                fontsize=14, fontweight="bold", color=tc, zorder=4)

    # ── No-RAG ────────────────────────────────────────────────────────────────
    ax = axes[0]
    tc, fc, ec = "#059669", "#D1FAE5", "#6EE7B7"

    node(ax, 0.5, 0.825, NW, NH, "Patient History Note",
         "#EFF6FF", "#93C5FD", fs=10.5)
    ann(ax, 0.825, "Free-text student note", "#2563EB")

    node(ax, 0.5, 0.700, NW, NH, "Concept Names",
         "#EFF6FF", "#93C5FD", fs=10.5)
    ann(ax, 0.700, "13–17 rubric concepts", "#2563EB")

    node(ax, 0.5, 0.545, NW, 0.11, "Granite-4.1-30b",
         "#D1FAE5", "#34D399", fs=12)
    ann(ax, 0.545, "1 API call / note", tc)

    node(ax, 0.5, 0.390, NW, NH, "Verdicts + Evidence",
         "#F3E8FF", "#C084FC", fs=10.5)
    ann(ax, 0.390, "present / absent + quote", "#7C3AED")

    node(ax, 0.5, 0.265, NW, NH, "Results Saved",
         "#F1F5F9", "#94A3B8", fs=10.5)

    arr(ax, 0.5, 0.778, 0.5, 0.748, tc)
    arr(ax, 0.5, 0.653, 0.5, 0.600, tc)
    arr(ax, 0.5, 0.490, 0.5, 0.437, tc)
    arr(ax, 0.5, 0.343, 0.5, 0.313, tc)

    # "No ChromaDB" badge
    badge = mpatches.FancyBboxPatch((0.12, 0.140), 0.76, 0.080,
        boxstyle="round,pad=0.02", facecolor="#F0FDF4", edgecolor=ec, linewidth=1.5, zorder=3)
    ax.add_patch(badge)
    ax.text(0.5, 0.180, "No retrieval — LLM uses\npre-trained medical knowledge",
            ha="center", va="center", fontsize=9, color=tc, fontweight="bold", zorder=4,
            multialignment="center")

    stat_footer(ax, ["F1 = 0.9423", "1 API call per note", "Fastest · Cheapest"], tc)

    # ── Naive RAG ─────────────────────────────────────────────────────────────
    ax = axes[1]
    tc, fc, ec = "#2563EB", "#DBEAFE", "#93C5FD"

    node(ax, 0.5, 0.825, NW, NH, "Patient History Note",
         "#EFF6FF", "#93C5FD", fs=10.5)
    ann(ax, 0.825, "Free-text student note", tc)

    node(ax, 0.5, 0.700, NW, NH, "ChromaDB Batch Query",
         "#FEF3C7", "#FCD34D", fs=10.5)
    ann(ax, 0.700, "Fetch all concepts (1 query)", "#B45309")

    node(ax, 0.5, 0.575, NW, NH, "Synonym Lists",
         "#FEF3C7", "#FCD34D", fs=10.5)
    ann(ax, 0.575, "All 13–17 synonym sets", "#B45309")

    node(ax, 0.5, 0.430, NW, 0.115, "Granite-4.1-30b",
         "#D1FAE5", "#34D399", fs=12)
    ann(ax, 0.430, "1 API call / note", "#059669")

    node(ax, 0.5, 0.275, NW, NH, "Verdicts + Evidence",
         "#F3E8FF", "#C084FC", fs=10.5)
    ann(ax, 0.275, "present / absent + quote", "#7C3AED")

    node(ax, 0.5, 0.155, NW, NH, "Results Saved",
         "#F1F5F9", "#94A3B8", fs=10.5)

    arr(ax, 0.5, 0.778, 0.5, 0.748, tc)
    arr(ax, 0.5, 0.653, 0.5, 0.623, tc)
    arr(ax, 0.5, 0.528, 0.5, 0.488, tc)
    arr(ax, 0.5, 0.373, 0.5, 0.323, tc)
    arr(ax, 0.5, 0.228, 0.5, 0.203, tc)

    stat_footer(ax, ["F1 = 0.9233", "1 ChromaDB query + 1 API call", "Synonym lists retrieved upfront"], tc)

    # ── Agentic RAG ───────────────────────────────────────────────────────────
    ax = axes[2]
    tc, fc, ec = "#DC2626", "#FEE2E2", "#FCA5A5"

    node(ax, 0.5, 0.865, NW, NH, "Patient History Note",
         "#EFF6FF", "#93C5FD", fs=10.5)
    ann(ax, 0.865, "Free-text student note", "#2563EB")

    node(ax, 0.5, 0.745, NW, NH, "Pydantic AI Agent",
         "#FEE2E2", "#FCA5A5", fs=10.5)
    ann(ax, 0.745, "Receives note + feature IDs", tc)

    # Dashed loop region
    loop = mpatches.FancyBboxPatch(
        (0.05, 0.355), 0.75, 0.335,
        boxstyle="round,pad=0.02",
        facecolor="#FFFBEB", edgecolor="#F59E0B",
        linewidth=1.8, linestyle=(0, (6, 3)), zorder=1
    )
    ax.add_patch(loop)
    ax.text(0.425, 0.670, "Repeated 13–17× per note",
            ha="center", va="center", fontsize=9,
            color="#B45309", style="italic", fontweight="bold", zorder=4)

    node(ax, 0.42, 0.590, 0.62, NH, "Tool: retrieve_concept_info()",
         "#FEF3C7", "#FCD34D", fs=9.5)

    node(ax, 0.42, 0.475, 0.62, NH, "ChromaDB Lookup",
         "#FEF3C7", "#FCD34D", fs=9.5)
    ann(ax, 0.475, "Synonyms returned", "#B45309")

    node(ax, 0.42, 0.365, 0.62, NH, "LLM: Verdict per Concept",
         "#D1FAE5", "#34D399", fs=9.5)
    ann(ax, 0.365, "1 API call per concept", "#059669")

    node(ax, 0.5, 0.235, NW, NH, "All Verdicts Collected",
         "#F3E8FF", "#C084FC", fs=10.5)
    ann(ax, 0.235, "After all concepts done", "#7C3AED")

    node(ax, 0.5, 0.120, NW, NH, "Results Saved",
         "#F1F5F9", "#94A3B8", fs=10.5)

    arr(ax, 0.5, 0.818, 0.5, 0.793, tc)
    arr(ax, 0.5, 0.698, 0.42, 0.638, tc)
    arr(ax, 0.42, 0.543, 0.42, 0.523, tc)
    arr(ax, 0.42, 0.428, 0.42, 0.413, tc)
    arr(ax, 0.42, 0.318, 0.5, 0.283, tc)
    arr(ax, 0.5, 0.188, 0.5, 0.168, tc)

    # Loop-back arrow on the right
    ax.annotate("", xy=(0.82, 0.590), xytext=(0.82, 0.365),
                arrowprops=dict(arrowstyle="-|>", color="#F59E0B",
                                lw=1.8, mutation_scale=14), zorder=2)
    ax.text(0.845, 0.478, "next\nconcept", fontsize=8.5,
            color="#B45309", ha="left", va="center",
            fontweight="bold", style="italic")

    stat_footer(ax, ["F1 = 0.7886", "13–17 API calls per note", "Full reasoning trace per concept"], tc)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96], w_pad=3)
    out = OUT_DIR / "pipeline_comparison.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Codebase map
# ─────────────────────────────────────────────────────────────────────────────

def make_codebase_map():
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor(C["bg"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    ax.set_title("ClinNoteRAG — Codebase Architecture Map",
                 fontsize=15, fontweight="bold", color="#1a252f", pad=14)

    def cbox(cx, cy, w, h, title, subtitle="", fc="#D6EAF8", ec="#2980B9", fs=8.5):
        r = mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.08",
            facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3
        )
        ax.add_patch(r)
        if subtitle:
            ax.text(cx, cy + 0.15, title, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color="#1a252f", zorder=4)
            ax.text(cx, cy - 0.18, subtitle, ha="center", va="center",
                    fontsize=fs - 1.5, color="#444", zorder=4,
                    style="italic", multialignment="center")
        else:
            ax.text(cx, cy, title, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color="#1a252f", zorder=4,
                    multialignment="center")

    def carr(x1, y1, x2, y2, lbl="", col="#5D6D7E"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                    lw=1.3, mutation_scale=13), zorder=2)
        if lbl:
            mx, my = (x1+x2)/2 + 0.1, (y1+y2)/2
            ax.text(mx, my, lbl, fontsize=7, color=col, style="italic")

    # ── Data layer ────────────────────────────────────────────────────────────
    ax.text(1.1, 9.5, "DATA", fontsize=9, fontweight="bold",
            color="#777", va="center")
    cbox(2.5, 9.0, 3.2, 0.7,
         "NBME_PN_HISTORY_FEATURES.txt",
         "143 rubric concepts | 10 cases",
         fc="#FEF9E7", ec="#B7950B")
    cbox(6.5, 9.0, 3.0, 0.7,
         "NBME_PN_HISTORY.txt",
         "43,985 patient notes",
         fc="#FEF9E7", ec="#B7950B")
    cbox(10.5, 9.0, 3.2, 0.7,
         "NBME_PN_HISTORY_ANNOTATIONS.txt",
         "34,660 expert physician labels",
         fc="#FEF9E7", ec="#B7950B")

    # ── Ingestion layer ───────────────────────────────────────────────────────
    ax.text(1.1, 7.8, "INGESTION", fontsize=9, fontweight="bold", color="#777")
    cbox(2.5, 7.3, 3.0, 0.8,
         "scripts/ingest_concepts.py",
         "Embeds concepts → ChromaDB\n(sentence-transformers)",
         fc=C["db"][0], ec=C["db"][1])
    cbox(10.5, 7.3, 2.6, 0.8,
         "services/embedder.py",
         "sentence-transformers\nall-MiniLM-L6-v2",
         fc=C["db"][0], ec=C["db"][1])

    # ── Vector DB ─────────────────────────────────────────────────────────────
    ax.text(1.1, 6.1, "VECTOR DB", fontsize=9, fontweight="bold", color="#777")
    cbox(3.5, 5.6, 2.8, 0.8,
         "ChromaDB",
         "nbme_concepts\n(synonym lists)",
         fc="#FDEBD0", ec="#E67E22", fs=9)

    # ── Services / RAG strategies ─────────────────────────────────────────────
    ax.text(1.1, 4.4, "RAG SERVICES", fontsize=9, fontweight="bold", color="#777")
    cbox(2.0, 3.8, 2.4, 0.9,
         "services/no_rag.py",
         "assess_note_no_rag()\n0 ChromaDB calls\n1 LLM call / note",
         fc=C["llm"][0], ec=C["llm"][1], fs=8)
    cbox(5.0, 3.8, 2.4, 0.9,
         "services/naive_rag.py",
         "assess_note_naive()\n1 ChromaDB batch\n1 LLM call / note",
         fc=C["llm"][0], ec=C["llm"][1], fs=8)
    cbox(8.5, 3.8, 2.4, 0.9,
         "services/agent.py",
         "assess_note()\n13-17 ChromaDB calls\n13-17 LLM calls / note",
         fc=C["agent"][0], ec=C["agent"][1], fs=8)

    # ── Evaluation + Web ──────────────────────────────────────────────────────
    ax.text(1.1, 2.6, "EVALUATION", fontsize=9, fontweight="bold", color="#777")
    cbox(4.5, 2.1, 3.4, 0.9,
         "scripts/evaluate.py",
         "--strategy [no_rag | naive_rag |\nagentic_rag]\nSaves results.csv + metrics.csv",
         fc=C["output"][0], ec=C["output"][1], fs=8)
    cbox(9.5, 2.1, 3.0, 0.9,
         "apps/assessment/ (Django)",
         "Web dashboard\nStudent submit + grader review\nAdmin panel",
         fc=C["input"][0], ec=C["input"][1], fs=8)

    # ── LLM ───────────────────────────────────────────────────────────────────
    cbox(13.5, 5.6, 2.4, 0.9,
         "CAIR LLM Server",
         "granite-4.1-30b\nvia LiteLLM\n(OpenAI-compatible)",
         fc="#F9EBEA", ec="#C0392B", fs=8)

    # ── Arrows ────────────────────────────────────────────────────────────────
    # Data → ingestion
    carr(2.5, 8.65, 2.5, 7.7)
    carr(10.5, 8.65, 10.5, 7.7)

    # ingest → chromadb
    carr(2.5, 6.9, 3.5, 6.0)

    # chromadb → services
    carr(3.5, 5.2, 2.0, 4.25)
    carr(3.5, 5.2, 5.0, 4.25)
    carr(3.5, 5.2, 8.5, 4.25)

    # services → evaluate
    carr(2.0, 3.35, 4.5, 2.55)
    carr(5.0, 3.35, 4.8, 2.55)
    carr(8.5, 3.35, 5.2, 2.55)

    # services → web
    carr(5.0, 3.35, 9.5, 2.55)

    # services → LLM
    carr(8.5, 3.35, 13.5, 5.15)

    plt.tight_layout()
    out = OUT_DIR / "codebase_map.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    print("Generating pipeline comparison diagram...")
    make_pipeline_comparison()
    print("Generating codebase map...")
    make_codebase_map()
    print("Done.")
