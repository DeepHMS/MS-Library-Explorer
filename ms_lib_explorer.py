# app.py
import io
import csv
import math
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import streamlit as st

st.set_page_config(page_title="PASEF Mapper", layout="wide")

# -------------------------
# Helpers
# -------------------------
def sniff_delimiter(file_bytes: bytes) -> str:
    # Look at the first kilobyte to guess delimiter
    head = file_bytes[:1024].decode(errors="ignore")
    return "," if head.count(",") >= head.count("\t") else "\t"

def read_table(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    if not raw:
        return pd.DataFrame()
    sep = sniff_delimiter(raw)
    return pd.read_csv(io.BytesIO(raw), sep=sep)

def parse_unimods(df: pd.DataFrame, selected_unimods: set | None = None):
    """Add UniMod_List and Has_Mod columns (selected_unimods may be empty)."""
    pattern = r"UniMod:(\d+)"
    df = df.copy()
    df["ModifiedPeptideSequence"] = df["ModifiedPeptideSequence"].fillna("")
    df["UniMod_List"] = df["ModifiedPeptideSequence"].apply(
        lambda x: tuple(sorted(re.findall(pattern, x)))
    )
    all_unis = sorted({u for x in df["UniMod_List"] for u in x})
    if selected_unimods is None:
        selected_unimods = set()
    df["Has_Mod"] = df["UniMod_List"].apply(lambda x: bool(set(x) & selected_unimods))
    return df, all_unis

def parse_pasef_windows(uploaded_txt, pasef_type: str):
    """Return (dia_windows, diag_windows) from a .txt file depending on pasef_type."""
    dia_windows, diag_windows = [], []
    if uploaded_txt is None:
        return dia_windows, diag_windows
    txt = uploaded_txt.read().decode(errors="ignore").splitlines()
    reader = csv.reader(txt)
    _ = next(reader, None)  # header
    for parts in reader:
        if not parts:
            continue
        if pasef_type == "DIA" and parts[0].strip() == "PASEF":
            # columns: type, ..., K0_low, K0_high, mz_low, mz_high
            # indices used below: (2,3,4,5)
            try:
                dia_windows.append(
                    (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
                )
            except Exception:
                continue
        elif pasef_type == "DIAGONAL" and parts[0].strip().lower() == "diagonal":
            # columns: "diagonal", k1, m1_start, m1_end, k2, m2_start
            try:
                diag_windows.append(
                    (float(parts[1]), float(parts[2]), float(parts[3]),
                     float(parts[4]), float(parts[5]))
                )
            except Exception:
                continue
    return dia_windows, diag_windows

def draw_dia(ax, dia_windows):
    for k1, k2, mz1, mz2 in dia_windows:
        ax.add_patch(Rectangle((mz1, k1), mz2 - mz1, k2 - k1,
                               edgecolor="red", facecolor="none",
                               linewidth=1.4, alpha=0.6, zorder=50))

def draw_diag(ax, diag_windows):
    for k1, m1s, m1e, k2, m2s in diag_windows:
        w = m1e - m1s
        pts = [(m1s, k1), (m1e, k1), (m2s + w, k2), (m2s, k2)]
        ax.add_patch(Polygon(pts, closed=True,
                             edgecolor="red", facecolor="none",
                             linewidth=1.4, alpha=0.6, zorder=50))

def draw_windows(ax, overlay, pasef_type, dia_windows, diag_windows):
    if not overlay:
        return
    if pasef_type == "DIA":
        draw_dia(ax, dia_windows)
    elif pasef_type == "DIAGONAL":
        draw_diag(ax, diag_windows)

def make_colormap(selected_unimods):
    base_colors = ["blue", "orange", "skyblue", "green", "purple", "gold", "brown"]
    return {u: base_colors[i % len(base_colors)] for i, u in enumerate(sorted(selected_unimods))}

def plot_panel(ax, data, title, xlim, ylim,
               show_nonuni: bool, show_unimod: bool,
               selected_unimods: set,
               overlay, pasef_type, dia_windows, diag_windows):
    data = data.copy()
    # Background (all or filtered)
    if show_nonuni:
        nonu = data[~data["Has_Mod"]] if "Has_Mod" in data.columns else data
        ax.scatter(nonu["PrecursorMz"], nonu["PrecursorIonMobility"],
                   color="grey", alpha=0.10, s=8, zorder=5)

    if show_unimod and selected_unimods:
        cmap = make_colormap(selected_unimods)
        for u in selected_unimods:
            sub = data[data["UniMod_List"].apply(lambda x: u in x)]
            if not sub.empty:
                ax.scatter(sub["PrecursorMz"], sub["PrecursorIonMobility"],
                           color=cmap[u], alpha=0.25, s=14, zorder=20, label=f"UniMod:{u}")

    draw_windows(ax, overlay, pasef_type, dia_windows, diag_windows)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Precursor m/z", fontsize=10)
    ax.set_ylabel("Ion Mobility (1/K0)", fontsize=10)

# -------------------------
# UI
# -------------------------
st.title("m/z vs Ion Mobility — PASEF Mapper (Streamlit)")

left, right = st.columns([1.2, 1])

with left:
    st.subheader("1) Upload Method Library")
    lib_file = st.file_uploader("Upload .csv or .tsv", type=["csv", "tsv"])
    req_cols = ["ProteinId", "PrecursorMz", "PeptideSequence",
                "ModifiedPeptideSequence", "PrecursorIonMobility", "PrecursorCharge"]

with right:
    st.subheader("2) Overlay PASEF Windows (Optional)")
    overlay = st.checkbox("Overlay PASEF windows", value=False)
    pasef_type = st.radio("PASEF type", ["DIA", "DIAGONAL"], horizontal=True, disabled=not overlay)
    win_file = st.file_uploader("Upload PASEF windows .txt", type=["txt"], disabled=not overlay)

st.divider()

c1, c2, c3, c4 = st.columns(4)
with c1:
    x_min = st.number_input("m/z min", value=0.0, step=50.0)
with c2:
    x_max = st.number_input("m/z max", value=1800.0, step=50.0)
with c3:
    y_min = st.number_input("1/K0 min", value=0.0, step=0.1, format="%.2f")
with c4:
    y_max = st.number_input("1/K0 max", value=1.90, step=0.05, format="%.2f")

xlim = (x_min, x_max)
ylim = (y_min, y_max)

st.subheader("3) UniMod Mapping & Layers")
map_unimod = st.checkbox("Enable UniMod parsing and highlighting", value=True)
show_nonuni = st.checkbox("Show non-UniMod layer (grey)", value=True)
show_unimod = st.checkbox("Show UniMod layer (colored)", value=True)
merge_uni = st.checkbox("Merge UniMod and non-UniMod together in same panel", value=True,
                        help="When ON (recommended), both layers render in the same panels. When OFF, only the selected layers render (still in same panels).")

st.info("Option 1 (Unified plot): Toggle UniMod & non-UniMod layers above to show either or both in the same panels.", icon="ℹ️")

# -------------------------
# Processing
# -------------------------
if lib_file is None:
    st.warning("Please upload a method library file to continue.")
    st.stop()

df = read_table(lib_file)
missing = [c for c in req_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Unique subsets for plotting
subset1 = ["ProteinId", "PrecursorMz", "PeptideSequence",
           "ModifiedPeptideSequence", "PrecursorIonMobility"]
df_all = df[subset1].drop_duplicates()

subset2 = subset1 + ["PrecursorCharge"]
df_chg = df[subset2].drop_duplicates()

# UniMod parse
selected_unimods = set()
all_unis = []
if map_unimod:
    df_all, all_unis = parse_unimods(df_all, None)
    df_chg, _ = parse_unimods(df_chg, None)

    # UniMod selector — default to ALL
    if all_unis:
        uni_sel = st.multiselect("Select UniMod types to highlight (default: ALL)", all_unis, default=all_unis)
        selected_unimods = set(uni_sel)
        # recompute Has_Mod using selection
        df_all["Has_Mod"] = df_all["UniMod_List"].apply(lambda x: bool(set(x) & selected_unimods))
        df_chg["Has_Mod"] = df_chg["UniMod_List"].apply(lambda x: bool(set(x) & selected_unimods))
    else:
        st.info("No UniMods detected in the library.")
        df_all["Has_Mod"] = False
        df_chg["Has_Mod"] = False
else:
    # still need columns for plotting logic
    df_all["UniMod_List"] = [[] for _ in range(len(df_all))]
    df_all["Has_Mod"] = False
    df_chg["UniMod_List"] = [[] for _ in range(len(df_chg))]
    df_chg["Has_Mod"] = False

# PASEF windows
dia_windows, diag_windows = [], []
if overlay:
    dia_windows, diag_windows = parse_pasef_windows(win_file, pasef_type)

# Charges for panels
charges = sorted(df_chg["PrecursorCharge"].dropna().unique())
# cap total panels to 6 like original layout (1 "All" + up to 5 charges)
charges_for_panels = charges[:5]
num_panels = 1 + len(charges_for_panels)
ncols = 3
nrows = math.ceil(num_panels / ncols)

# -------------------------
# Plotting
# -------------------------
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
axes = axes.flatten() if isinstance(axes, (list, tuple, np.ndarray)) else [axes]

# Panel 1: All precursors (Unique)
ax0 = axes[0]
title0 = "All Precursors (Unique)"
plot_panel(ax0, df_all, title0, xlim, ylim,
           show_nonuni=show_nonuni or merge_uni,
           show_unimod=show_unimod or merge_uni,
           selected_unimods=selected_unimods,
           overlay=overlay, pasef_type=pasef_type,
           dia_windows=dia_windows, diag_windows=diag_windows)

# Charge-specific panels
for i, ch in enumerate(charges_for_panels, start=1):
    ax = axes[i]
    ch_data = df_chg[df_chg["PrecursorCharge"] == ch]
    title = f"Charge = {int(ch)}"
    plot_panel(ax, ch_data, title, xlim, ylim,
               show_nonuni=show_nonuni or merge_uni,
               show_unimod=show_unimod or merge_uni,
               selected_unimods=selected_unimods,
               overlay=overlay, pasef_type=pasef_type,
               dia_windows=dia_windows, diag_windows=diag_windows)

# Hide any unused axes
for j in range(1 + len(charges_for_panels), len(axes)):
    axes[j].axis("off")

suptitle = "m/z vs Ion Mobility"
if map_unimod: suptitle += " — UniMod Highlighting"
if overlay: suptitle += f" — {pasef_type} PASEF Windows"
fig.suptitle(suptitle, fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])

st.pyplot(fig, clear_figure=True)

# -------------------------
# Downloads & Summary
# -------------------------
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
st.download_button("Download figure (PNG)", data=buf.getvalue(), file_name="pasef_mapper.png", mime="image/png")

st.caption(f"Detected charges: {', '.join(map(lambda x: str(int(x)), charges)) if len(charges) else 'None'}")
st.caption(f"Detected UniMods: {', '.join(all_unis) if all_unis else 'None'}")
