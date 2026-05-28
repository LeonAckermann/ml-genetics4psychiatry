"""
Interactive visualization of significant SNP-feature associations.

Shows:
  1. Distribution of significant features per SNP (conditioned on feature p-value threshold)
  2. Top features ranked by number of significant SNPs

SNPs are filtered by the p-value derived from their GWAS Z-score (column 2),
converted via the two-tailed standard normal: p = 2 × Φ(−|Z|).
Feature significance is determined by the p-value matrix (sampled_p/).
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy.stats as stats
import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Discover available (illness, sampling_threshold) combinations
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
P_DIR = os.path.join(SCRIPT_DIR, "data/sampled/low")

def _scan_files():
    """Return dict: illness → sorted list of sampling-threshold strings."""
    available = {}
    for path in glob.glob(os.path.join(P_DIR, "sampled_*_p*.txt")):
        name = os.path.basename(path)  # e.g. sampled_SCZ_p0.1.txt
        # skip derived files (_significant_*)
        if "_significant" in name:
            continue
        parts = name.replace(".txt", "").split("_")
        # parts: ['sampled', illness, 'p0.1']  (illness may be multi-word but isn't here)
        illness = parts[1]
        thresh  = parts[2]   # e.g. 'p0.1'
        available.setdefault(illness, []).append(thresh)
    for ill in available:
        available[ill] = sorted(available[ill], key=lambda x: float(x[1:]))
    return available

AVAILABLE = _scan_files()
ILLNESSES  = sorted(AVAILABLE.keys())

# ---------------------------------------------------------------------------
# Per-file data cache  {(illness, thresh_str) → (feat_arr, gwas_pvals, feature_names)}
# ---------------------------------------------------------------------------
_cache: dict = {}

def load_dataset(illness: str, thresh_str: str):
    key = (illness, thresh_str)
    if key in _cache:
        return _cache[key]

    p_path = os.path.join(P_DIR, f"sampled_{illness}_{thresh_str}.txt")
    print(f"Loading {p_path} …")
    df = pd.read_csv(p_path, sep="\t", index_col=0)

    gwas_z      = df["Z"].copy()
    feature_df  = df.drop(columns=["Z"])
    feat_names  = feature_df.columns.tolist()
    # Convert feature Z-scores → two-tailed p-values on the fly
    feat_z_arr  = feature_df.values.astype(np.float32)
    feat_arr    = (2 * stats.norm.sf(np.abs(feat_z_arr))).astype(np.float32)
    gwas_pvals  = pd.Series(
        2 * stats.norm.sf(np.abs(gwas_z.values)),
        index=gwas_z.index,
    )
    print(f"  → {feat_arr.shape[0]:,} SNPs × {feat_arr.shape[1]:,} features")
    _cache[key] = (feat_arr, gwas_pvals, feat_names)
    return _cache[key]

# Pre-load the default (first illness, last threshold = most data)
DEFAULT_ILLNESS = "SCZ" if "SCZ" in ILLNESSES else ILLNESSES[0]
DEFAULT_THRESH  = AVAILABLE[DEFAULT_ILLNESS][-1]
load_dataset(DEFAULT_ILLNESS, DEFAULT_THRESH)

# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, title="SNP–Feature Significance Explorer")

PVAL_MARKS = {
    -5: "1e-5", -4: "1e-4", -3: "0.001",
    -2: "0.01",  -1: "0.05",
    round(np.log10(0.1), 4): "0.1",
    round(np.log10(0.5), 4): "0.5",
}

ILLNESS_COLORS = {
    "SCZ":  "#4a90d9",
    "BIP":  "#e05c5c",
    "ADHD": "#f5a623",
    "MDD":  "#7b68ee",
    "OCD":  "#5bb86e",
    "AZ":   "#d4a017",
}

def illness_options():
    return [{"label": ill, "value": ill} for ill in ILLNESSES]

def thresh_options(illness):
    return [{"label": t, "value": t} for t in AVAILABLE.get(illness, [])]

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "padding": "20px",
           "maxWidth": "1400px", "margin": "0 auto"},
    children=[
        html.H2("SNP–Feature Significance Explorer", style={"marginBottom": "4px"}),
        html.P(
            "Select an illness and adjust thresholds to explore how many brain-imaging "
            "features are significantly associated with each SNP (and vice versa). "
            "SNPs are pre-filtered by their GWAS Z-score significance "
            "(p = 2·Φ(−|Z|), two-tailed standard normal).",
            style={"color": "#555", "marginTop": 0, "fontSize": "13px"},
        ),

        # ---- Illness + sampling threshold selectors ----
        html.Div(
            style={"display": "flex", "gap": "30px", "alignItems": "flex-end",
                   "marginBottom": "24px", "flexWrap": "wrap"},
            children=[
                html.Div([
                    html.Label("Illness", style={"fontWeight": "bold", "display": "block", "marginBottom": "6px"}),
                    dcc.Dropdown(
                        id="illness-select",
                        options=illness_options(),
                        value=DEFAULT_ILLNESS,
                        clearable=False,
                        style={"width": "160px"},
                    ),
                ]),
                html.Div([
                    html.Label("Sampling threshold", style={"fontWeight": "bold", "display": "block", "marginBottom": "6px"}),
                    html.P("Max GWAS p used during data sampling",
                           style={"color": "#777", "fontSize": "12px", "margin": "0 0 4px 0"}),
                    dcc.Dropdown(
                        id="samp-thresh-select",
                        options=thresh_options(DEFAULT_ILLNESS),
                        value=DEFAULT_THRESH,
                        clearable=False,
                        style={"width": "120px"},
                    ),
                ]),
                html.Div(id="dataset-info", style={
                    "fontSize": "13px", "color": "#555",
                    "paddingBottom": "6px", "alignSelf": "flex-end",
                }),
            ],
        ),

        # ---- Threshold sliders ----
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                   "gap": "30px", "marginBottom": "20px"},
            children=[
                html.Div([
                    html.Label("GWAS Z-score p-value threshold (SNP filter)",
                               style={"fontWeight": "bold"}),
                    html.P("Only SNPs with GWAS p ≤ this value are included.",
                           style={"color": "#777", "fontSize": "13px", "marginTop": 0}),
                    dcc.Slider(
                        id="gwas-thresh", min=-5, max=0, step=0.05,
                        value=-2, marks=PVAL_MARKS,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ]),
                html.Div([
                    html.Label("Feature p-value threshold (significance cutoff)",
                               style={"fontWeight": "bold"}),
                    html.P("A SNP–feature pair is significant when feature p ≤ this value.",
                           style={"color": "#777", "fontSize": "13px", "marginTop": 0}),
                    dcc.Slider(
                        id="feat-thresh", min=-5, max=0, step=0.05,
                        value=-2, marks=PVAL_MARKS,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ]),
            ],
        ),

        html.Div([
            html.Label("Top-N features to display", style={"fontWeight": "bold"}),
            dcc.Slider(id="top-n", min=10, max=100, step=5, value=30,
                       marks={v: str(v) for v in [10, 20, 30, 50, 75, 100]},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"marginBottom": "20px", "maxWidth": "600px"}),

        # ---- Stats bar ----
        html.Div(id="stats-box", style={
            "background": "#f0f4ff", "borderRadius": "8px",
            "padding": "12px 20px", "marginBottom": "20px",
            "fontSize": "14px", "color": "#333",
        }),

        # ---- Plots ----
        dcc.Loading(dcc.Graph(id="main-plot",    style={"height": "520px"})),
        dcc.Loading(dcc.Graph(id="feature-plot", style={"height": "520px"})),

        # Shared computed counts store
        dcc.Store(id="counts-store"),
    ],
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("samp-thresh-select", "options"),
    Output("samp-thresh-select", "value"),
    Input("illness-select", "value"),
)
def update_thresh_options(illness):
    opts = thresh_options(illness)
    # default to last option (most data)
    default = opts[-1]["value"] if opts else None
    return opts, default


@callback(
    Output("dataset-info", "children"),
    Input("illness-select", "value"),
    Input("samp-thresh-select", "value"),
)
def update_dataset_info(illness, samp_thresh):
    if not illness or not samp_thresh:
        return ""
    feat_arr, _, _ = load_dataset(illness, samp_thresh)
    return f"{feat_arr.shape[0]:,} SNPs · {feat_arr.shape[1]:,} features loaded"


@callback(
    Output("counts-store", "data"),
    Output("stats-box", "children"),
    Input("illness-select", "value"),
    Input("samp-thresh-select", "value"),
    Input("gwas-thresh", "value"),
    Input("feat-thresh", "value"),
)
def compute_counts(illness, samp_thresh, log_gwas_thresh, log_feat_thresh):
    if not illness or not samp_thresh:
        return {}, "Select an illness to begin."

    gwas_thresh = 10 ** log_gwas_thresh
    feat_thresh = 10 ** log_feat_thresh

    feat_arr, gwas_pvals, feat_names = load_dataset(illness, samp_thresh)
    n_features = feat_arr.shape[1]

    mask      = gwas_pvals.values <= gwas_thresh
    n_filtered = int(mask.sum())

    if n_filtered == 0:
        return (
            {"sig_per_snp": [], "sig_per_feat": [0] * n_features,
             "illness": illness, "n_features": n_features},
            html.Span(
                f"No SNPs pass the GWAS threshold (p ≤ {gwas_thresh:.0e}) — loosen the SNP filter.",
                style={"color": "red"},
            ),
        )

    arr_sub    = feat_arr[mask]
    sig_matrix = arr_sub <= feat_thresh

    sig_per_snp  = sig_matrix.sum(axis=1).tolist()
    sig_per_feat = sig_matrix.sum(axis=0).tolist()
    total_sig    = int(sig_matrix.sum())
    pct          = 100 * total_sig / (n_filtered * n_features)

    summary = [
        html.Strong(f"{illness}"),
        f"  ·  ",
        html.Strong(f"{n_filtered:,}"),
        f" SNPs (GWAS p ≤ {gwas_thresh:.0e})  ·  ",
        html.Strong(f"{total_sig:,}"),
        f" significant pairs ({pct:.3f}% of submatrix)  ·  ",
        f"feature p ≤ {feat_thresh:.0e}",
    ]
    return (
        {"sig_per_snp": sig_per_snp, "sig_per_feat": sig_per_feat,
         "illness": illness, "n_features": n_features},
        summary,
    )


@callback(
    Output("main-plot", "figure"),
    Input("counts-store", "data"),
    Input("gwas-thresh", "value"),
    Input("feat-thresh", "value"),
)
def update_main_plot(counts, log_gwas_thresh, log_feat_thresh):
    if not counts or not counts.get("sig_per_snp"):
        fig = go.Figure()
        fig.add_annotation(text="No SNPs pass current filter", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    illness     = counts.get("illness", "")
    sig_per_snp = np.array(counts["sig_per_snp"])
    gwas_thresh = 10 ** log_gwas_thresh
    feat_thresh = 10 ** log_feat_thresh
    color       = ILLNESS_COLORS.get(illness, "#4a90d9")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Significant features per SNP (histogram)",
            "SNPs with ≥ k significant features (survival curve)",
        ],
        horizontal_spacing=0.12,
    )

    # Left: histogram
    fig.add_trace(
        go.Histogram(
            x=sig_per_snp,
            nbinsx=min(int(sig_per_snp.max()) + 1 if len(sig_per_snp) else 1, 120),
            marker_color=color,
            hovertemplate="Sig. features: %{x}<br>SNPs: %{y}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Right: survival / CCDF
    sorted_vals = np.sort(sig_per_snp)[::-1]
    fig.add_trace(
        go.Scatter(
            x=sorted_vals, y=np.arange(len(sorted_vals)),
            mode="lines", line=dict(color=color, width=2),
            hovertemplate="≥%{x} sig. features → %{y} SNPs<extra></extra>",
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="# significant features",         row=1, col=1)
    fig.update_yaxes(title_text="# SNPs",                         row=1, col=1)
    fig.update_xaxes(title_text="k (min. significant features)",  row=1, col=2)
    fig.update_yaxes(title_text="# SNPs with ≥ k",               row=1, col=2)
    fig.update_layout(
        title=f"{illness} · GWAS p ≤ {gwas_thresh:.0e} · feature p ≤ {feat_thresh:.0e}",
        showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


@callback(
    Output("feature-plot", "figure"),
    Input("counts-store", "data"),
    Input("top-n", "value"),
    Input("gwas-thresh", "value"),
    Input("feat-thresh", "value"),
    Input("illness-select", "value"),
    Input("samp-thresh-select", "value"),
)
def update_feature_plot(counts, top_n, log_gwas_thresh, log_feat_thresh, illness, samp_thresh):
    if not counts or not counts.get("sig_per_feat"):
        return go.Figure().add_annotation(text="No data", showarrow=False,
                                          xref="paper", yref="paper", x=0.5, y=0.5)

    _, _, feat_names = load_dataset(illness, samp_thresh)
    sig_per_feat = np.array(counts["sig_per_feat"])
    gwas_thresh  = 10 ** log_gwas_thresh
    feat_thresh  = 10 ** log_feat_thresh
    color        = ILLNESS_COLORS.get(illness, "#5bb86e")

    top_idx    = np.argsort(sig_per_feat)[::-1][:top_n]
    top_counts = sig_per_feat[top_idx]
    top_labels = [feat_names[i] for i in top_idx]

    fig = go.Figure(go.Bar(
        x=top_counts, y=top_labels,
        orientation="h",
        marker_color=color,
        hovertemplate="%{y}<br>Significant SNPs: %{x}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Top {top_n} features by # significant SNPs "
              f"({illness} · GWAS p ≤ {gwas_thresh:.0e} · feature p ≤ {feat_thresh:.0e})",
        xaxis_title="# significant SNPs",
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white",
        height=max(520, top_n * 16),
        margin=dict(l=260),
    )
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nOpen http://127.0.0.1:8050 in your browser\n")
    app.run(debug=False, port=8050)
