#############################
# PASEF MAPPER STREAMLIT #
# With Google Drive + Local Upload
#############################
import io
import re
import csv
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

# ---------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(page_title="PASEF Mapper", layout="wide")

# ---------------------------------------------------------
# Helper: Auto-detect delimiter
# ---------------------------------------------------------
def autodetect_sep(head_bytes: bytes) -> str:
    text = head_bytes.decode(errors="ignore")
    return "," if text.count(",") >= text.count("\t") else "\t"

# ---------------------------------------------------------
# Parse Google Drive shareable link → File ID
# ---------------------------------------------------------
def extract_drive_file_id(url: str):
    m = re.search(r"/d/([^/]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"id=([^&]+)", url)
    if m:
        return m.group(1)
    return None

# ---------------------------------------------------------
# Download from Google Drive (Handles Virus Scan Warning)
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def download_from_google_drive(file_id: str) -> bytes:
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(URL, stream=True)
    
    # Handle virus scan warning
    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break
    
    if confirm_token:
        response = session.get(URL + f"&confirm={confirm_token}", stream=True)
    
    if response.status_code != 200:
        raise RuntimeError(f"Download failed (status {response.status_code}). Check link & permissions.")
    
    return response.content

# ---------------------------------------------------------
# PASEF / UniMod helper logic
# ---------------------------------------------------------
def parse_unimods(df, selected_unimods=None):
    pattern = r"UniMod:(\d+)"
    df = df.copy()
    df["ModifiedPeptideSequence"] = df["ModifiedPeptideSequence"].fillna("")
    df["UniMod_List"] = df["ModifiedPeptideSequence"].apply(
        lambda x: tuple(sorted(set(re.findall(pattern, x))))
    )
    all_unis = sorted({u for x in df["UniMod_List"] for u in x})
    
    if selected_unimods is None:
        # Default: top 3 most frequent
        top3 = (df["UniMod_List"].explode().value_counts().head(3).index.tolist())
        selected_unimods = set(top3) if top3 else set()
    
    df["Has_Mod"] = df["UniMod_List"].apply(lambda x: bool(set(x) & selected_unimods))
    return df, all_unis

def parse_pasef_windows_txt_bytes(txt_bytes: bytes, pasef_type: str):
    dia_windows, diag_windows = [], []
    lines = txt_bytes.decode(errors="ignore").splitlines()
    reader = csv.reader(lines)
    next(reader, None)  # header
    for parts in reader:
        if not parts:
            continue
        if pasef_type == "DIA" and parts[0].strip() == "PASEF":
            try:
                dia_windows.append((float(parts[2]), float(parts[3]),
                                    float(parts[4]), float(parts[5])))
            except:
                continue
        if pasef_type == "DIAGONAL" and parts[0].strip().lower() == "diagonal":
            try:
                diag_windows.append((float(parts[1]), float(parts[2]), float(parts[3]),
                                     float(parts[4]), float(parts[5])))
            except:
                continue
    return dia_windows, diag_windows

def draw_dia(ax, dia_windows):
    for k1, k2, mz1, mz2 in dia_windows:
        ax.add_patch(Rectangle((mz1, k1), mz2-mz1, k2-k1,
                               edgecolor="red", facecolor="none",
                               linewidth=1.4, alpha=0.6))

def draw_diag(ax, diag_windows):
    for k1, m1s, m1e, k2, m2s in diag_windows:
        w = m1e - m1s
        poly = [(m1s, k1), (m1e, k1), (m2s+w, k2), (m2s, k2)]
        ax.add_patch(Polygon(poly, closed=True,
                             edgecolor="red", facecolor="none",
                             linewidth=1.4, alpha=0.6))

def draw_windows(ax, overlay, pasef_type, dia_windows, diag_windows):
    if not overlay:
        return
    if pasef_type == "DIA":
        draw_dia(ax, dia_windows)
    else:
        draw_diag(ax, diag_windows)

def make_colormap(selected_unimods):
    base_colors = ["blue","orange","green","purple","gold","cyan","brown","pink","gray","olive"]
    return {u: base_colors[i % len(base_colors)] for i,u in enumerate(sorted(selected_unimods))}

def plot_panel(ax, data, title, xlim, ylim,
               show_nonuni, show_unimod, selected_unimods,
               overlay, pasef_type, dia_windows, diag_windows):
    data = data.copy()
    if show_nonuni:
        nonu = data[~data["Has_Mod"]]
        ax.scatter(nonu["PrecursorMz"], nonu["PrecursorIonMobility"],
                   color="grey", alpha=0.12, s=8)
    if show_unimod:
        cmap = make_colormap(selected_unimods)
        for u in selected_unimods:
            sub = data[data["UniMod_List"].apply(lambda x: u in x)]
            if len(sub) > 0:
                ax.scatter(sub["PrecursorMz"], sub["PrecursorIonMobility"],
                           color=cmap[u], alpha=0.32, s=14, label=f"UniMod:{u}")
    draw_windows(ax, overlay, pasef_type, dia_windows, diag_windows)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Precursor m/z")
    ax.set_ylabel("Ion Mobility (1/K0)")

# ---------------------------------------------------------
# UI: File Input Method Selection
# ---------------------------------------------------------
st.title("PASEF Mapper — Visualize DIA-PASEF Libraries")
st.write("Upload local file **< 200 MB** or use **Google Drive shareable link** for larger files.")

input_method = st.radio(
    "Choose input method:",
    ["Local Upload (< 200 MB)", "Google Drive (any size)"],
    horizontal=True
)

df = None
MAX_UPLOAD_MB = 200
dtypes = {
    "ProteinId": "category",
    "PrecursorMz": "float32",
    "PeptideSequence": "string",
    "ModifiedPeptideSequence": "string",
    "PrecursorIonMobility": "float32",
    "PrecursorCharge": "Int8",
}

# -------------------------------
# 1. Local Upload
# -------------------------------
if input_method == "Local Upload (< 200 MB)":
    lib_file = st.file_uploader("Upload Library (.csv / .tsv / .txt)", type=["csv","tsv","txt"])
    if lib_file is not None:
        size_mb = lib_file.size / (1024*1024)
        if size_mb > MAX_UPLOAD_MB:
            st.error(f"File too large ({size_mb:.1f} MB). Use Google Drive method.")
            st.stop()
        st.success(f"Uploaded ({size_mb:.1f} MB)")
        raw = lib_file.read()
        sep = autodetect_sep(raw[:2000])
        df = pd.read_csv(io.BytesIO(raw), sep=sep, dtype=dtypes, engine='c')
        st.write("Preview:", df.head())

# -------------------------------
# 2. Google Drive
# -------------------------------
else:
    gd_link = st.text_input("Paste Google Drive **shareable link** (Anyone with link → Viewer)")
    if gd_link:
        file_id = extract_drive_file_id(gd_link)
        if not file_id:
            st.error("Invalid Google Drive link. Use format: `.../d/FILE_ID/...` or `?id=FILE_ID`")
            st.stop()
        with st.spinner("Downloading from Google Drive..."):
            try:
                file_bytes = download_from_google_drive(file_id)
            except Exception as e:
                st.error(f"Download failed: {e}")
                st.stop()
            sep = autodetect_sep(file_bytes[:2000])
            df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, dtype=dtypes, engine='c')
        st.success("Library loaded from Google Drive")
        st.write("Preview:", df.head())

# Stop if no data
if df is None:
    st.info("Please upload a library or provide a Google Drive link.")
    st.stop()

# ---------------------------------------------------------
# Validate required columns
# ---------------------------------------------------------
req_cols = ["ProteinId","PrecursorMz","PeptideSequence",
            "ModifiedPeptideSequence","PrecursorIonMobility","PrecursorCharge"]
missing = [c for c in req_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ---------------------------------------------------------
# Optional PASEF windows
# ---------------------------------------------------------
st.header("Optional: Overlay PASEF Windows")
overlay = st.checkbox("Overlay PASEF Windows", value=False)
pasef_type = st.radio("PASEF Type", ["DIA","DIAGONAL"], horizontal=True, disabled=not overlay)
dia_windows, diag_windows = [], []

if overlay:
    wopt = st.radio("Windows file source:", ["Upload (.txt)", "Google Drive"], horizontal=True)
    if wopt == "Upload (.txt)":
        win_file = st.file_uploader("Upload windows file (.txt)", type=["txt"], key="win_local")
        if win_file:
            w_bytes = win_file.read()
            dia_windows, diag_windows = parse_pasef_windows_txt_bytes(w_bytes, pasef_type)
            st.success("Windows file loaded.")
    else:
        win_link = st.text_input("Google Drive link (windows .txt)", key="win_drive")
        if win_link:
            fid = extract_drive_file_id(win_link)
            if fid:
                w_bytes = download_from_google_drive(fid)
                dia_windows, diag_windows = parse_pasef_windows_txt_bytes(w_bytes, pasef_type)
                st.success("Windows loaded from Drive")

# ---------------------------------------------------------
# Plotting Controls
# ---------------------------------------------------------
st.header("Plot Controls")
c1,c2,c3,c4 = st.columns(4)
with c1: x_min = st.number_input("m/z min", value=0.0, step=10.0)
with c2: x_max = st.number_input("m/z max", value=1800.0, step=10.0)
with c3: y_min = st.number_input("1/K0 min", value=0.0, step=0.1)
with c4: y_max = st.number_input("1/K0 max", value=1.9, step=0.1)
xlim = (x_min, x_max)
ylim = (y_min, y_max)

map_unimod = st.checkbox("Enable UniMod Mapping", value=True)
show_nonuni = st.checkbox("Show Non-UniMod", value=True)
show_unimod = st.checkbox("Show UniMod", value=True)

# ---------------------------------------------------------
# Prepare data subsets
# ---------------------------------------------------------
sub1 = ["ProteinId","PrecursorMz","PeptideSequence",
        "ModifiedPeptideSequence","PrecursorIonMobility"]
df_all = df[sub1].drop_duplicates()
sub2 = sub1 + ["PrecursorCharge"]
df_chg = df[sub2].drop_duplicates()

selected_unimods = set()
all_unis = []

if map_unimod:
    df_all, all_unis = parse_unimods(df_all)
    df_chg, _ = parse_unimods(df_chg)
    if all_unis:
        selected_unimods = set(st.multiselect("Select UniMods", all_unis, default=list(selected_unimods)))
    df_all["Has_Mod"] = df_all["UniMod_List"].apply(lambda x: bool(set(x) & selected_unimods))
    df_chg["Has_Mod"] = df_chg["UniMod_List"].apply(lambda x: bool(set(x) & selected_unimods))
else:
    df_all["Has_Mod"] = False
    df_chg["Has_Mod"] = False

# ---------------------------------------------------------
# Generate Plot
# ---------------------------------------------------------
charges = sorted(df_chg["PrecursorCharge"].dropna().unique().astype(int))
charges_for_panels = [int(c) for c in charges[:5]]
n_panels = 1 + len(charges_for_panels)
ncols = min(3, n_panels)
nrows = math.ceil(n_panels / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), constrained_layout=True)
axes = np.array(axes).flatten()[:n_panels]  # Only use needed axes

# All precursors
plot_panel(axes[0], df_all, "All Precursors", xlim, ylim,
           show_nonuni, show_unimod, selected_unimods,
           overlay, pasef_type, dia_windows, diag_windows)

# Charge panels
for i, ch in enumerate(charges_for_panels, start=1):
    ch_data = df_chg[df_chg["PrecursorCharge"] == ch]
    plot_panel(axes[i], ch_data, f"Charge +{ch}", xlim, ylim,
               show_nonuni, show_unimod, selected_unimods,
               overlay, pasef_type, dia_windows, diag_windows)

# Hide unused
for ax in axes[n_panels:]:
    ax.axis("off")

fig.suptitle("Precursor m/z vs Ion Mobility (1/K₀)", fontsize=18)
st.pyplot(fig)
plt.close(fig)  # Prevent memory leak

# ---------------------------------------------------------
# Download Button
# ---------------------------------------------------------
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
buf.seek(0)
st.download_button(
    "Download Plot as PNG",
    buf.getvalue(),
    file_name="pasef_mapper_plot.png",
    mime="image/png"
)
