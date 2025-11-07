# app.py
import io
import os
import csv
import math
import re
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

import streamlit as st

# Google OAuth / Drive
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import Flow
from googleapiclient.http import MediaIoBaseDownload

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="PASEF Mapper (Google Drive OAuth)", layout="wide")

# -----------------------------
# OAuth / Google Drive helpers
# -----------------------------
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def get_client_config_from_secrets():
    """
    Expect secrets to contain:
      [oauth_client]
      client_id="..."
      client_secret="..."
      redirect_uri="https://<your-app-url>/"  # MUST match Cloud console
    """
    cc = st.secrets["oauth_client"]
    client_config = {
        "web": {
            "client_id": cc["client_id"],
            "project_id": cc.get("project_id", "streamlit-oauth"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": cc["client_secret"],
            "redirect_uris": [cc["redirect_uri"]],
            "javascript_origins": [cc["redirect_uri"].rstrip("/")],
        }
    }
    return client_config

def build_authorize_url():
    client_config = get_client_config_from_secrets()
    flow = Flow.from_client_config(client_config=client_config, scopes=SCOPES)
    flow.redirect_uri = client_config["web"]["redirect_uris"][0]
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent"
    )
    st.session_state["oauth_state"] = state
    return auth_url

def exchange_code_for_token(auth_code: str):
    client_config = get_client_config_from_secrets()
    flow = Flow.from_client_config(client_config=client_config, scopes=SCOPES, state=st.session_state.get("oauth_state"))
    flow.redirect_uri = client_config["web"]["redirect_uris"][0]
    flow.fetch_token(code=auth_code)
    creds = flow.credentials
    st.session_state["google_creds"] = {
        "token": creds.token,
        "refresh_token": getattr(creds, "refresh_token", None),
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }

def get_drive_service():
    data = st.session_state.get("google_creds")
    if not data:
        return None
    creds = Credentials(
        token=data["token"],
        refresh_token=data.get("refresh_token"),
        token_uri=data["token_uri"],
        client_id=data["client_id"],
        client_secret=data["client_secret"],
        scopes=data["scopes"],
    )
    return build("drive", "v3", credentials=creds)

def parse_query_params_for_code():
    # Handle redirect from Google (code in URL)
    params = st.query_params
    if "code" in params:
        return params["code"]
    return None

# -----------------------------
# Drive file utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def drive_download_bytes(file_id: str, meta_fields="id,name,size,mimeType") -> tuple[bytes, dict]:
    """
    Chunked download from Google Drive for large files.
    Returns (content_bytes, metadata_dict)
    """
    service = get_drive_service()
    if service is None:
        raise RuntimeError("Not authenticated with Google. Please sign in.")

    meta = service.files().get(fileId=file_id, fields=meta_fields).execute()
    request = service.files().get_media(fileId=file_id)

    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request, chunksize=10 * 1024 * 1024)  # 10 MB chunks
    done = False
    while not done:
        status, done = downloader.next_chunk()
        # You can print progress to server logs if desired.

    return buffer.getvalue(), meta

def autodetect_sep(head_bytes: bytes) -> str:
    head = head_bytes.decode(errors="ignore")
    return "," if head.count(",") >= head.count("\t") else "\t"

def sniff_sep(content: bytes, user_sep: str | None) -> str:
    if user_sep in (",", "\t"):
        return user_sep
    return autodetect_sep(content[:2000])

def read_table_from_bytes(content: bytes, sep: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content), sep=sep)

# -----------------------------
# PASEF / UniMod logic
# -----------------------------
def parse_unimods(df: pd.DataFrame, selected_unimods: set | None = None):
    pattern = r"UniMod:(\d+)"
    df = df.copy()
    df["ModifiedPeptideSequence"] = df["ModifiedPeptideSequence"].fillna("")
    df["UniMod_List"] = df["ModifiedPeptideSequence"].apply(
        lambda x: tuple(sorted(re.findall(pattern, x)))
    )
    all_unis = sorted({u for x in df["UniMod_List"] for u in x})
    if selected_unimods is None:
        selected_unimods = set(all_unis)
    df["Has_Mod"] = df["UniMod_List"].apply(lambda x: bool(set(x) & selected_unimods))
    return df, all_unis

def parse_pasef_windows_txt_bytes(txt_bytes: bytes, pasef_type: str):
    dia_windows, diag_windows = [], []
    lines = txt_bytes.decode(errors="ignore").splitlines()
    reader = csv.reader(lines)
    _ = next(reader, None)  # header
    for parts in reader:
        if not parts:
            continue
        if pasef_type == "DIA" and parts[0].strip() == "PASEF":
            try:
                dia_windows.append((float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])))
            except Exception:
                continue
        elif pasef_type == "DIAGONAL" and parts[0].strip().lower() == "diagonal":
            try:
                diag_windows.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])))
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

# -----------------------------
# UI: Title and Auth
# -----------------------------
st.title("m/z vs Ion Mobility — PASEF Mapper (Google Drive OAuth per-user)")

# 1) Handle OAuth redirect code (if present)
code_in_url = parse_query_params_for_code()
if code_in_url and "google_creds" not in st.session_state:
    try:
        exchange_code_for_token(code_in_url)
        st.success("✅ Google sign-in complete.")
        # Clean the URL query params
        st.query_params.clear()
    except Exception as e:
        st.error(f"OAuth error: {e}")

# 2) Auth / Sign-in UI
if "google_creds" not in st.session_state:
    st.info("Sign in with Google to access **your own Google Drive** files securely.")
    auth_url = build_authorize_url()
    st.link_button("🔐 Sign in with Google", auth_url, type="primary")
    st.stop()
else:
    st.success("Signed in with Google Drive ✅")

# Build Drive service
service = get_drive_service()
if service is None:
    st.error("Could not initialize Google Drive service. Please sign in again.")
    st.stop()

# -----------------------------
# Controls: Load Library from Drive
# -----------------------------
st.header("1) Load Method Library (from your Google Drive)")
help_txt = "Paste a Google Drive File ID (from a share link) for your large library TSV/CSV."
lib_col1, lib_col2 = st.columns([2,1])
with lib_col1:
    lib_file_id = st.text_input("Google Drive File ID (library)", placeholder="e.g., 1AbCDEFghiJKLmnopQRsTuvWxYz", help=help_txt)
with lib_col2:
    sep_choice = st.selectbox("Delimiter", options=["Auto", "Comma (,)", "Tab (\\t)"], index=0)

if lib_file_id:
    with st.spinner("Downloading library from Drive..."):
        try:
            lib_bytes, lib_meta = drive_download_bytes(lib_file_id)
        except Exception as e:
            st.error(f"Failed to download library file: {e}")
            st.stop()

    sep = None
    if sep_choice == "Comma (,)":
        sep = ","
    elif sep_choice == "Tab (\\t)":
        sep = "\t"

    sep = sniff_sep(lib_bytes, sep)
    try:
        df = read_table_from_bytes(lib_bytes, sep)
    except Exception as e:
        st.error(f"Failed to parse library as CSV/TSV: {e}")
        st.stop()

    st.success(f"Library loaded: **{lib_meta.get('name','file')}** — rows: {df.shape[0]:,}, cols: {df.shape[1]}")
    with st.expander("Preview (head)"):
        st.dataframe(df.head())
else:
    st.warning("Enter a Google Drive File ID for the library to continue.")
    st.stop()

# -----------------------------
# Required columns check
# -----------------------------
req_cols = ["ProteinId", "PrecursorMz", "PeptideSequence",
            "ModifiedPeptideSequence", "PrecursorIonMobility", "PrecursorCharge"]
missing = [c for c in req_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# -----------------------------
# PASEF Windows (optional)
# -----------------------------
st.header("2) Optional: Overlay PASEF Windows")

overlay = st.checkbox("Overlay PASEF windows", value=False)
pasef_type = st.radio("PASEF type", ["DIA", "DIAGONAL"], horizontal=True, disabled=not overlay)

dia_windows, diag_windows = [], []
if overlay:
    wopt = st.radio("Provide window file via", ["Google Drive (File ID)", "Local upload"], horizontal=True)
    if wopt == "Google Drive (File ID)":
        win_id = st.text_input("Google Drive File ID (PASEF windows .txt)")
        if win_id:
            with st.spinner("Downloading PASEF windows from Drive..."):
                try:
                    w_bytes, w_meta = drive_download_bytes(win_id)
                except Exception as e:
                    st.error(f"Failed to download PASEF windows: {e}")
                    st.stop()
            dia_windows, diag_windows = parse_pasef_windows_txt_bytes(w_bytes, pasef_type)
            st.success(f"Loaded window file: **{w_meta.get('name','file')}**")
    else:
        up = st.file_uploader("Upload PASEF windows .txt", type=["txt"])
        if up:
            w_bytes = up.read()
            dia_windows, diag_windows = parse_pasef_windows_txt_bytes(w_bytes, pasef_type)
            st.success(f"Uploaded window file: **{up.name}**")

# -----------------------------
# Axes and UniMod Controls
# -----------------------------
st.header("3) Plot Controls")

c1, c2, c3, c4 = st.columns(4)
with c1:
    x_min = st.number_input("m/z min", value=0.0, step=50.0)
with c2:
    x_max = st.number_input("m/z max", value=1800.0, step=50.0)
with c3:
    y_min = st.number_input("1/K0 min", value=0.0, step=0.1, format="%.2f")
with c4:
    y_max = st.number_input("1/K0 max", value=1.90, step=0.05, format="%.2f")
xlim = (x_min, x_max); ylim = (y_min, y_max)

map_unimod = st.checkbox("Enable UniMod parsing and highlighting", value=True)
show_nonuni = st.checkbox("Show non-UniMod layer (grey)", value=True)
show_unimod = st.checkbox("Show UniMod layer (colored)", value=True)
merge_uni = st.checkbox("Merge UniMod & non-UniMod in same panels (unified view)", value=True,
                        help="Unified Option 1: both layers in the same panels. Turn OFF to view either layer alone (still unified panels).")

# -----------------------------
# Prepare data subsets
# -----------------------------
subset1 = ["ProteinId", "PrecursorMz", "PeptideSequence",
           "ModifiedPeptideSequence", "PrecursorIonMobility"]
df_all = df[subset1].drop_duplicates()

subset2 = subset1 + ["PrecursorCharge"]
df_chg = df[subset2].drop_duplicates()

selected_unimods = set()
all_unis = []
if map_unimod:
    df_all, all_unis = parse_unimods(df_all, None)
    df_chg, _ = parse_unimods(df_chg, None)
    if all_unis:
        uni_sel = st.multiselect("Select UniMod types to highlight (default: ALL)", all_unis, default=all_unis)
        selected_unimods = set(uni_sel)
        df_all["Has_Mod"] = df_all["UniMod_List"].apply(lambda x: bool(set(x) & selected_unimods))
        df_chg["Has_Mod"] = df_chg["UniMod_List"].apply(lambda x: bool(set(x) & selected_unimods))
    else:
        st.info("No UniMods detected in the library.")
        df_all["Has_Mod"] = False
        df_chg["Has_Mod"] = False
else:
    df_all["UniMod_List"] = [[] for _ in range(len(df_all))]
    df_all["Has_Mod"] = False
    df_chg["UniMod_List"] = [[] for _ in range(len(df_chg))]
    df_chg["Has_Mod"] = False

# Identify charges; limit to up to 5 to keep 1+5=6 panels like your original
charges = sorted(df_chg["PrecursorCharge"].dropna().unique())
charges_for_panels = charges[:5]
num_panels = 1 + len(charges_for_panels)
ncols = 3
nrows = math.ceil(num_panels / ncols)

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])

# Panel 1: All
ax0 = axes[0]
plot_panel(ax0, df_all, "All Precursors (Unique)", xlim, ylim,
           show_nonuni=show_nonuni or merge_uni,
           show_unimod=show_unimod or merge_uni,
           selected_unimods=selected_unimods,
           overlay=overlay, pasef_type=pasef_type,
           dia_windows=dia_windows, diag_windows=diag_windows)

# Charge panels
for i, ch in enumerate(charges_for_panels, start=1):
    ax = axes[i]
    ch_data = df_chg[df_chg["PrecursorCharge"] == ch]
    plot_panel(ax, ch_data, f"Charge = {int(ch)}", xlim, ylim,
               show_nonuni=show_nonuni or merge_uni,
               show_unimod=show_unimod or merge_uni,
               selected_unimods=selected_unimods,
               overlay=overlay, pasef_type=pasef_type,
               dia_windows=dia_windows, diag_windows=diag_windows)

# Hide extra axes
for j in range(1 + len(charges_for_panels), len(axes)):
    axes[j].axis("off")

suptitle = "m/z vs Ion Mobility"
if map_unimod: suptitle += " — UniMod Highlighting"
if overlay: suptitle += f" — {pasef_type} PASEF Windows"
fig.suptitle(suptitle, fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])

st.pyplot(fig, clear_figure=True)

# Download button
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
st.download_button("Download figure (PNG)", data=buf.getvalue(),
                   file_name="pasef_mapper.png", mime="image/png")

# Summary
st.caption(f"Detected charges: {', '.join(map(lambda x: str(int(x)), charges)) if len(charges) else 'None'}")
st.caption(f"Detected UniMods: {', '.join(all_unis) if all_unis else 'None'}")
