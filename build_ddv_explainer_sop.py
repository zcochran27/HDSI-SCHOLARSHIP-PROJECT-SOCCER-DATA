"""Build the SOP-aware interactive DDV explainer.

Emits two files:
    ddv_explainer_sop.html   — page skeleton + CSS, loads the JS below
    ddv_explainer_sop.js     — precomputed data (base64 Float32 arrays) and
                                all interactive logic

Run:
    python train_ddv_sop_models.py
    python build_ddv_explainer_sop.py
"""

import base64
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).parent
MODELS = ROOT / "ddv_sop_models"
PARQUET = ROOT / "data" / "defensive_duels_preprocessed.parquet"
OUT_HTML = ROOT / "ddv_explainer_sop.html"
OUT_JS = ROOT / "ddv_explainer_sop.js"

GRID_NX = 31   # 31 x-cells in [0, 60]
GRID_NY = 21   # 21 y-cells in [0, 80]
SOP_NS = 5     # 5 slider positions per speed-of-play feature (must match sop_meta quantiles)

# User-facing labels for the three speed-of-play features.
SOP_LABELS = {
    "delta_x_30s": "Ball x-progression (last 30s)",
    "dist_30s": "Ball distance travelled (last 30s)",
    "dist_ratio_30v60": "Tempo ratio (last 30s / 60s)",
}


# -------------------- load + precompute --------------------

def load_models():
    return (
        joblib.load(MODELS / "outcome_model.joblib"),
        joblib.load(MODELS / "goal_cond_model.joblib"),
        joblib.load(MODELS / "label_encoder.joblib"),
        np.load(MODELS / "population_values.npy"),
        joblib.load(MODELS / "sop_meta.joblib"),
    )


def compute_surfaces(outcome_model, goal_cond_model, class_names, sop_meta):
    """Returns arrays of shape (N_combos, K, G) where G = GRID_NX*GRID_NY,
    K = n_classes, N_combos = SOP_NS**3. Layout matches what the JS expects."""
    gx = np.linspace(0, 60, GRID_NX)
    gy = np.linspace(0, 80, GRID_NY)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    xy = np.column_stack([GX.ravel(), GY.ravel()])  # (G, 2), row-major over (ix, iy)
    G = len(xy)
    K = len(class_names)

    sop_cols = sop_meta["cols"]
    sop_positions = [sop_meta["quantiles"][c] for c in sop_cols]
    for col, pos in zip(sop_cols, sop_positions):
        assert len(pos) == SOP_NS, (
            f"sop_meta quantiles for {col} has {len(pos)} entries, expected {SOP_NS}"
        )

    N = SOP_NS ** 3
    p_outcome_all = np.zeros((N, K, G), dtype=np.float32)
    p_goal_all = np.zeros((N, K, G), dtype=np.float32)

    combo_idx = 0
    for i in range(SOP_NS):
        for j in range(SOP_NS):
            for k in range(SOP_NS):
                X = np.column_stack([
                    xy,
                    np.full(G, sop_positions[0][i]),
                    np.full(G, sop_positions[1][j]),
                    np.full(G, sop_positions[2][k]),
                ])
                p_out = outcome_model.predict_proba(X)  # (G, K)
                p_goal = np.column_stack([
                    goal_cond_model.predict_proba(
                        np.column_stack([X, np.full(G, c)])
                    )[:, 1]
                    for c in range(K)
                ])  # (G, K)
                # Store as (K, G): transpose
                p_outcome_all[combo_idx] = p_out.T
                p_goal_all[combo_idx] = p_goal.T
                combo_idx += 1

    return p_outcome_all, p_goal_all, sop_positions


# -------------------- duel density P(duel | x, y) on the same grid --------------------

def compute_duel_density(sop_cols):
    """Smoothed marginal duel density on the GRID_NX × GRID_NY grid.

    Layout is flattened as ix*ny + iy to match surface arrays so the JS can
    multiply element-wise with DDV cells.
    """
    duels = pd.read_parquet(PARQUET)
    d = duels[duels.is_defensive_duel & (duels.start_x < 60)].copy()
    d = d.dropna(subset=sop_cols)
    x_edges = np.linspace(0, 60, GRID_NX + 1)
    y_edges = np.linspace(0, 80, GRID_NY + 1)
    H, _, _ = np.histogram2d(
        d["start_x"].to_numpy(),
        d["start_y"].to_numpy(),
        bins=[x_edges, y_edges],
    )  # shape (GRID_NX, GRID_NY) indexed [ix, iy]
    H = gaussian_filter(H.astype(np.float64), sigma=2.0)
    s = H.sum()
    if s > 0:
        H /= s
    return H.astype(np.float32), int(len(d))


# -------------------- histogram (static PNG, same as original) --------------------

def render_histogram(values):
    lo, hi = np.percentile(values, [0.5, 99.5])
    fig_w_in, fig_h_in = 10.0, 2.6
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    fig.patch.set_facecolor("#18222a")
    left_frac, right_frac = 0.06, 0.985
    bottom_frac, top_frac = 0.30, 0.84
    ax = fig.add_axes([left_frac, bottom_frac, right_frac - left_frac, top_frac - bottom_frac])
    ax.set_facecolor("#18222a")
    ax.hist(np.clip(values, lo, hi), bins=120, color="#60a5fa", alpha=0.85, edgecolor="none")
    ax.axvline(0, color="#8fa3b0", lw=1, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Duel value (goal probability averted relative to expectation)",
                  color="#cfdae2", fontsize=10)
    ax.set_ylabel("count", color="#cfdae2", fontsize=10)
    ax.tick_params(colors="#8fa3b0", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#2a3b48")
    ax.set_title(f"Distribution of DDV across {len(values):,} defensive duels",
                 color="#e7eef3", fontsize=12, pad=8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140, facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    png = "data:image/png;base64," + base64.b64encode(buf.read()).decode()
    return png, float(lo), float(hi), float(left_frac), float(right_frac)


# -------------------- payload --------------------

def encode_f32(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).ravel(order="C").tobytes()).decode()


def build_payload():
    outcome_model, goal_cond_model, le, values, sop_meta = load_models()
    class_names = list(le.classes_)

    p_outcome_all, p_goal_all, sop_positions = compute_surfaces(
        outcome_model, goal_cond_model, class_names, sop_meta
    )

    sop_cols = sop_meta["cols"]
    p_duel_grid, n_duels_for_density = compute_duel_density(sop_cols)

    hist_png, hist_lo, hist_hi, hist_left_frac, hist_right_frac = render_histogram(values)
    percentiles = [float(q) for q in np.percentile(values, np.arange(0, 101))]

    return {
        "class_names": class_names,
        "sop_cols": sop_cols,
        "sop_labels": {c: SOP_LABELS.get(c, c) for c in sop_cols},
        "sop_positions": {c: sop_positions[i] for i, c in enumerate(sop_cols)},
        "sop_stats": sop_meta["stats"],
        "grid": {
            "nx": GRID_NX, "ny": GRID_NY,
            "x_min": 0.0, "x_max": 60.0,
            "y_min": 0.0, "y_max": 80.0,
            "sop_ns": SOP_NS,
            "n_combos": SOP_NS ** 3,
            "n_classes": len(class_names),
            # Layout: [combo][class][flat_xy], flat_xy = ix*ny + iy
            "p_outcome_b64": encode_f32(p_outcome_all),
            "p_goal_b64":    encode_f32(p_goal_all),
            # Marginal duel density P(duel | x, y), shape (nx, ny) → flat ix*ny+iy
            "p_duel_b64":    encode_f32(p_duel_grid),
            "p_duel_max":    float(p_duel_grid.max()),
            "p_duel_n_duels": n_duels_for_density,
            "p_goal_global_max": float(p_goal_all.max()),
            "p_outcome_global_max": float(p_outcome_all.max()),
        },
        "percentiles": percentiles,
        "population": {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "n": int(len(values)),
            "hist_lo": hist_lo, "hist_hi": hist_hi,
            "hist_left_frac": hist_left_frac, "hist_right_frac": hist_right_frac,
        },
        "images": {"histogram": hist_png},
    }


# -------------------- HTML + JS templates --------------------

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Defensive Duel Value — Speed of Play Explainer</title>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<style>
:root {
  --bg: #0f1418;
  --panel: #18222a;
  --panel2: #1f2d36;
  --line: #2a3b48;
  --fg: #e7eef3;
  --muted: #8fa3b0;
  --accent: #6ee7b7;
  --red: #f87171;
  --blue: #60a5fa;
  --amber: #fbbf24;
  --green: #6ee7b7;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--fg); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; }
body { padding: 24px 36px 80px; max-width: 1400px; margin: 0 auto; }
h1 { font-weight: 700; font-size: 28px; margin: 0 0 6px; }
h2 { font-weight: 600; font-size: 18px; margin: 24px 0 10px; color: var(--fg); }
h3 { font-weight: 600; font-size: 13px; margin: 12px 0 6px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
p { line-height: 1.55; color: #cfdae2; }
small, .muted { color: var(--muted); }
code { background: #0b1116; padding: 1px 5px; border-radius: 3px; font-size: 13px; color: #b5e3d0; }

.grid-2 { display: grid; gap: 16px; grid-template-columns: 1.15fr 1fr; align-items: stretch; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; }
.card.pitch-card { padding: 14px; }

#pitch-wrap { position: relative; width: 100%; user-select: none; }
#pitch-wrap svg { display: block; width: 100%; height: auto; }

.outcome-buttons { display: flex; gap: 8px; margin: 10px 0 0; flex-wrap: wrap; }
.outcome-btn {
  flex: 1; min-width: 120px; padding: 10px 14px; border-radius: 8px;
  border: 1px solid var(--line); background: #0b1116; color: var(--fg);
  cursor: pointer; font-size: 14px; transition: all 120ms ease; font-weight: 500;
}
.outcome-btn:hover { background: #162129; }
.outcome-btn.selected { border-width: 2px; padding: 9px 13px; }
.outcome-btn.selected.recovered { border-color: var(--green); background: rgba(110,231,183,0.14); color: var(--green); }
.outcome-btn.selected.stopped { border-color: var(--amber); background: rgba(251,191,36,0.14); color: var(--amber); }
.outcome-btn.selected.beat { border-color: var(--red); background: rgba(248,113,113,0.14); color: var(--red); }

.slider-group { display: flex; flex-direction: column; gap: 10px; margin-top: 6px; }
.slider-row { display: grid; grid-template-columns: 160px 1fr 72px; align-items: center; gap: 10px; font-size: 12.5px; }
.slider-row .lbl { color: var(--fg); font-weight: 500; }
.slider-row input[type=range] {
  width: 100%; accent-color: var(--accent); height: 4px; background: transparent;
}
.slider-row .val { font-family: 'SFMono-Regular', Menlo, Consolas, monospace; font-size: 12px; color: var(--fg); text-align: right; }
.slider-row .ticks { color: var(--muted); font-size: 10px; display: flex; justify-content: space-between; grid-column: 2; }
.slider-row .sub { color: var(--muted); font-size: 11px; margin-top: 1px; }

.bar-row { display: grid; grid-template-columns: 120px 1fr 82px; align-items: center; gap: 8px; margin: 5px 0; font-size: 13px; }
.bar-row .label { color: var(--muted); }
.bar-track { background: #0b1116; border: 1px solid var(--line); border-radius: 3px; height: 20px; position: relative; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 2px; transition: width 220ms ease; }
.bar-row .value { text-align: right; font-variant-numeric: tabular-nums; color: var(--fg); font-family: 'SFMono-Regular', Menlo, Consolas, monospace; }
.bar-row.selected .label { color: var(--fg); font-weight: 600; }
.bar-row.selected .bar-track { box-shadow: 0 0 0 1px var(--accent); }

.formula-block { background: var(--panel2); border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; font-family: 'SFMono-Regular', Menlo, Consolas, monospace; font-size: 12.5px; line-height: 1.75; overflow-x: auto; }
.formula-block .sym { color: var(--muted); }
.formula-block .val { color: var(--fg); font-weight: 600; }
.formula-block .pos { color: var(--green); }
.formula-block .neg { color: var(--red); }

.final { margin-top: 10px; font-size: 17px; display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
.final .value-box { font-family: 'SFMono-Regular', Menlo, Consolas, monospace; font-size: 26px; font-weight: 700; padding: 6px 14px; border-radius: 6px; letter-spacing: 0.5px; }

.stat-strip { display: flex; gap: 18px; margin: 8px 0 0; flex-wrap: wrap; font-size: 13px; }
.stat-strip .stat { display: flex; flex-direction: column; gap: 2px; }
.stat-strip .stat .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-strip .stat .v { font-family: 'SFMono-Regular', Menlo, Consolas, monospace; font-size: 15px; font-weight: 600; }

.hist-wrap { position: relative; margin-top: 8px; }
.hist-wrap img { width: 100%; border-radius: 8px; display: block; }
.hist-wrap .marker { position: absolute; top: 0; bottom: 38px; width: 2px; background: #ff4d5a; pointer-events: none; }
.hist-wrap .marker-cap { position: absolute; top: 0; transform: translateX(-50%); color: #ff4d5a; font-size: 11px; background: var(--panel); padding: 2px 6px; border-radius: 3px; border: 1px solid rgba(255,77,90,0.4); white-space: nowrap; font-weight: 600; }

.surface-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 10px; }
.surface-panel { background: #0b1116; border: 1px solid var(--line); border-radius: 8px; padding: 8px; }
.surface-panel canvas { display: block; width: 100%; height: auto; background: #1b2a1f; border-radius: 4px; }
.surface-title { font-size: 12px; color: var(--muted); margin: 2px 0 4px; font-family: 'SFMono-Regular', Menlo, Consolas, monospace; }
.colorbar { display: flex; align-items: center; gap: 6px; font-size: 10px; color: var(--muted); margin-top: 4px; }
.colorbar .bar { flex: 1; height: 8px; border-radius: 2px; border: 1px solid var(--line); }

.tabs { display: flex; gap: 4px; margin: 10px 0 4px; border-bottom: 1px solid var(--line); flex-wrap: wrap; }
.tab { padding: 7px 13px; cursor: pointer; font-size: 13px; color: var(--muted); border-bottom: 2px solid transparent; }
.tab.active { color: var(--fg); border-bottom-color: var(--accent); }
.tab:hover { color: var(--fg); }

.empty-hint { color: var(--muted); font-size: 13px; font-style: italic; padding: 28px 0; text-align: center; }
details { margin: 10px 0; }
summary { cursor: pointer; color: var(--muted); font-size: 13px; padding: 4px 0; }
summary:hover { color: var(--fg); }
.formula-math { font-family: 'SFMono-Regular', Menlo, Consolas, monospace; padding: 10px 14px; background: #0b1116; border-radius: 6px; border: 1px solid var(--line); font-size: 13px; }

.badge { font-size: 12px; padding: 3px 8px; border-radius: 999px; background: #0b1116; border: 1px solid var(--line); color: var(--muted); display: inline-block; }
.badge.recovered { background: rgba(110,231,183,0.12); color: var(--green); border-color: rgba(110,231,183,0.3); }
.badge.stopped { background: rgba(251,191,36,0.12); color: var(--amber); border-color: rgba(251,191,36,0.3); }
.badge.beat { background: rgba(248,113,113,0.12); color: var(--red); border-color: rgba(248,113,113,0.3); }

@media (max-width: 960px) {
  .grid-2 { grid-template-columns: 1fr; }
  .surface-row { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<h1>Defensive Duel Value — Speed of Play Explainer</h1>
<p class="muted" style="max-width:920px">
  Click anywhere on the defensive half to place a duel, pick the outcome, and adjust the three speed-of-play context
  sliders to see how the location prior <code>P(outcome | x, y, speed&nbsp;of&nbsp;play)</code> and the conditional
  goal risk <code>P(goal | x, y, outcome, speed&nbsp;of&nbsp;play)</code> respond. The <b>Duel Value</b> is the gap
  between expected and realized danger, both evaluated at the same location and speed-of-play context.
</p>

<div class="grid-2">
  <div class="card pitch-card">
    <h3>Step 1 · click to place the duel</h3>
    <div id="pitch-wrap"></div>
    <h3 style="margin-top:12px">Step 2 · pick the observed outcome</h3>
    <div class="outcome-buttons">
      <button class="outcome-btn recovered" data-outcome="recovered">recovered</button>
      <button class="outcome-btn stopped" data-outcome="stopped">stopped</button>
      <button class="outcome-btn beat" data-outcome="beat">beat</button>
    </div>
    <h3 style="margin-top:14px">Step 3 · Speed of play context</h3>
    <div class="slider-group" id="sop-sliders-main"></div>
    <div class="stat-strip">
      <div class="stat"><span class="k">x</span><span class="v" id="meta-x">—</span></div>
      <div class="stat"><span class="k">y</span><span class="v" id="meta-y">—</span></div>
      <div class="stat"><span class="k">outcome</span><span class="v" id="meta-outcome">—</span></div>
    </div>
  </div>

  <div class="card" id="breakdown-card">
    <div id="empty-state" class="empty-hint">
      Click on the pitch and pick an outcome to see the DDV breakdown.
    </div>
    <div id="breakdown" style="display:none">
      <h3>P(outcome | x, y, speed&nbsp;of&nbsp;play)</h3>
      <p class="muted" style="margin:4px 0 8px;font-size:13px">
        Bars shift when you move the speed-of-play sliders — how often each outcome occurs at this location given this
        context.
      </p>
      <div id="bars-outcome"></div>

      <h3 style="margin-top:16px">P(goal | x, y, outcome, speed&nbsp;of&nbsp;play)</h3>
      <p class="muted" style="margin:4px 0 8px;font-size:13px">
        Chance a goal is conceded in the next 20&nbsp;s given the location, outcome, and speed-of-play context.
      </p>
      <div id="bars-goal"></div>

      <h3 style="margin-top:16px">Combining the two</h3>
      <div class="formula-block" id="formula-block"></div>
      <div class="final">
        <span class="muted">Duel value</span>
        <span class="value-box" id="final-value">—</span>
        <span class="muted" id="final-explain"></span>
      </div>
    </div>
  </div>
</div>

<div class="card" style="margin-top:16px">
  <h2>Where does this duel rank?</h2>
  <p class="muted" style="margin:4px 0 8px">
    Histogram of duel values across all <span id="total-n"></span> defensive duels in the dataset. The red line marks
    this duel's value; percentile is the share of duels with a smaller value.
  </p>
  <div class="stat-strip">
    <div class="stat"><span class="k">duel value</span><span class="v" id="pct-val">—</span></div>
    <div class="stat"><span class="k">percentile</span><span class="v" id="pct-pct">—</span></div>
    <div class="stat"><span class="k">population mean</span><span class="v" id="pct-mean">—</span></div>
    <div class="stat"><span class="k">population std</span><span class="v" id="pct-std">—</span></div>
  </div>
  <div class="hist-wrap">
    <img id="hist-img" alt="DDV histogram" />
    <div id="hist-marker" class="marker" style="display:none"></div>
    <div id="hist-marker-cap" class="marker-cap" style="display:none">—</div>
  </div>
</div>

<div class="card" style="margin-top:16px">
  <h2>How the probabilities vary across the pitch</h2>
  <p class="muted" style="margin:4px 0 8px">
    The speed-of-play sliders below control the context used by <em>all four</em> surface tabs. Each tab shows one
    panel per outcome (beat / recovered / stopped). Toggle tabs to see where the models place each quantity across the
    defensive half. The last tab weights DDV by where duels actually occur — highlighting locations where typical
    dangerous duels happen.
  </p>

  <h3 style="margin-top:8px">Speed of play context for surfaces</h3>
  <div class="slider-group" id="sop-sliders-surface"></div>

  <div class="tabs">
    <div class="tab active" data-tab="outcome">P(outcome | x, y, speed&nbsp;of&nbsp;play)</div>
    <div class="tab" data-tab="goal">P(goal | x, y, outcome, speed&nbsp;of&nbsp;play)</div>
    <div class="tab" data-tab="ddv">DDV | x, y, outcome, speed&nbsp;of&nbsp;play</div>
    <div class="tab" data-tab="ddv_duel">DDV × P(duel | x, y)</div>
  </div>
  <div class="surface-row" id="surface-row"></div>
</div>

<details style="margin-top:18px">
  <summary>Formula derivation</summary>
  <div class="formula-math" style="margin-top:8px">
    <p>Expected danger is the marginal goal probability at this location given the speed-of-play context:</p>
    <pre style="white-space:pre-wrap">expected = Σ_c  p(c | x, y, speed of play) · p(goal | c, x, y, speed of play)</pre>
    <p>Realized danger is the goal probability under the actual observed outcome c*:</p>
    <pre style="white-space:pre-wrap">realized = p(goal | c*, x, y, speed of play)</pre>
    <p>Defender value is positive when the observed outcome beat expectations:</p>
    <pre style="white-space:pre-wrap">DDV = expected − realized</pre>
    <p>The final tab additionally weights DDV by duel density, so the visualization emphasises
    locations where dangerous duels <em>typically</em> happen:</p>
    <pre style="white-space:pre-wrap">DDV × P(duel | x, y)</pre>
  </div>
</details>

<script src="ddv_explainer_sop.js"></script>
</body>
</html>
"""


JS_TEMPLATE = r"""// ddv_explainer_sop.js — generated by build_ddv_explainer_sop.py.
// All precomputed data is embedded in DATA; pitch surfaces are decoded from
// base64 Float32 arrays and rendered to canvases on slider/tab updates.

const DATA = /*__DATA__*/ {};

const CLS_COLOR = {
  "recovered": "#6ee7b7",
  "stopped":   "#fbbf24",
  "beat":      "#f87171",
};

// ===== Base64 -> Float32Array =====
function b64ToF32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

const G_SIZE = DATA.grid.nx * DATA.grid.ny;
const K = DATA.grid.n_classes;
const P_OUTCOME = b64ToF32(DATA.grid.p_outcome_b64);   // (N_combos, K, G)
const P_GOAL    = b64ToF32(DATA.grid.p_goal_b64);      // (N_combos, K, G)
const P_DUEL    = b64ToF32(DATA.grid.p_duel_b64);      // (G,) marginal duel density
const SN = DATA.grid.sop_ns;
const SOP_LABELS = DATA.sop_labels;

function comboIdx(i, j, k) { return (i * SN + j) * SN + k; }

// Read the K-long probability vector at (combo, flat_xy) — strides across class
function readOutcomeVec(combo, flat_xy) {
  const base = combo * K * G_SIZE;
  const out = new Array(K);
  for (let c = 0; c < K; c++) out[c] = P_OUTCOME[base + c * G_SIZE + flat_xy];
  return out;
}
function readGoalVec(combo, flat_xy) {
  const base = combo * K * G_SIZE;
  const out = new Array(K);
  for (let c = 0; c < K; c++) out[c] = P_GOAL[base + c * G_SIZE + flat_xy];
  return out;
}

// Return a Float32Array view (length G_SIZE) for a single class surface (contiguous).
function sliceSurface(arr, combo, classIdx) {
  const offset = combo * K * G_SIZE + classIdx * G_SIZE;
  return arr.subarray(offset, offset + G_SIZE);
}

// ===== State =====
const state = {
  x: null, y: null, outcome: null,
  sopMain: [2, 2, 2],      // slider indices per SOP feature (default = median)
  sopSurface: [2, 2, 2],
  surfaceTab: "outcome",
};

// ===== Colormaps =====
const VIRIDIS = [
  [68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],
  [31,158,137],[53,183,121],[110,206,88],[181,222,43],[253,231,37]
];
const MAGMA = [
  [0,0,4],[28,16,68],[79,18,123],[129,37,129],[181,54,122],
  [229,80,100],[251,135,97],[254,194,135],[252,253,191]
];
function lerp(a, b, t) { return a + (b - a) * t; }
function sampleCmap(cmap, t) {
  if (!isFinite(t)) return [40,40,40];
  t = Math.max(0, Math.min(1, t));
  const s = t * (cmap.length - 1);
  const i = Math.floor(s);
  const f = s - i;
  const a = cmap[i], b = cmap[Math.min(cmap.length - 1, i + 1)];
  return [lerp(a[0], b[0], f) | 0, lerp(a[1], b[1], f) | 0, lerp(a[2], b[2], f) | 0];
}
function viridis(t) { return sampleCmap(VIRIDIS, t); }
function magma(t)   { return sampleCmap(MAGMA, t); }
function diverging(t) {
  // t in [-1, 1]; 0 -> near-panel bg, negative -> red, positive -> green
  if (!isFinite(t)) return [40,40,40];
  t = Math.max(-1, Math.min(1, t));
  if (t >= 0) {
    // green
    return [
      lerp(27, 110, t) | 0,
      lerp(42, 231, t) | 0,
      lerp(31, 183, t) | 0,
    ];
  } else {
    return [
      lerp(27, 248, -t) | 0,
      lerp(42, 113, -t) | 0,
      lerp(31, 113, -t) | 0,
    ];
  }
}

// ===== Pitch SVG =====
const svgNS = "http://www.w3.org/2000/svg";
const PITCH_W = 120, PITCH_H = 80;
const X_MAX_CLICK = 60;
const pitchWrap = document.getElementById("pitch-wrap");

function drawPitchLines(svg) {
  const stroke = "rgba(255,255,255,0.55)";
  const sw = 0.3;
  const line = (x1,y1,x2,y2) => {
    const e = document.createElementNS(svgNS, "line");
    e.setAttribute("x1", x1); e.setAttribute("y1", y1);
    e.setAttribute("x2", x2); e.setAttribute("y2", y2);
    e.setAttribute("stroke", stroke); e.setAttribute("stroke-width", sw);
    svg.appendChild(e);
  };
  const rect = (x,y,w,h) => {
    const e = document.createElementNS(svgNS, "rect");
    e.setAttribute("x", x); e.setAttribute("y", y);
    e.setAttribute("width", w); e.setAttribute("height", h);
    e.setAttribute("fill", "none");
    e.setAttribute("stroke", stroke); e.setAttribute("stroke-width", sw);
    svg.appendChild(e);
  };
  const circ = (cx,cy,r) => {
    const e = document.createElementNS(svgNS, "circle");
    e.setAttribute("cx", cx); e.setAttribute("cy", cy);
    e.setAttribute("r", r);
    e.setAttribute("fill", "none");
    e.setAttribute("stroke", stroke); e.setAttribute("stroke-width", sw);
    svg.appendChild(e);
  };
  rect(0,0,120,80); line(60,0,60,80); circ(60,40,9.15);
  rect(0,18,18,44); rect(102,18,18,44);
  rect(0,30,6,20);  rect(114,30,6,20);
  const spot = document.createElementNS(svgNS, "circle");
  spot.setAttribute("cx", 60); spot.setAttribute("cy", 40);
  spot.setAttribute("r", 0.4); spot.setAttribute("fill", stroke);
  svg.appendChild(spot);
  [[12,40],[108,40]].forEach(([cx,cy]) => {
    const e = document.createElementNS(svgNS, "circle");
    e.setAttribute("cx", cx); e.setAttribute("cy", cy);
    e.setAttribute("r", 0.4); e.setAttribute("fill", stroke);
    svg.appendChild(e);
  });
}

function buildPitch() {
  pitchWrap.innerHTML = "";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${PITCH_W} ${PITCH_H}`);
  svg.style.cursor = "crosshair";
  svg.style.background = "#1b2a1f";
  svg.style.borderRadius = "8px";

  const shade = document.createElementNS(svgNS, "rect");
  shade.setAttribute("x", 0); shade.setAttribute("y", 0);
  shade.setAttribute("width", 60); shade.setAttribute("height", 80);
  shade.setAttribute("fill", "#223"); shade.setAttribute("opacity", 0.35);
  svg.appendChild(shade);

  drawPitchLines(svg);

  const dl = document.createElementNS(svgNS, "text");
  dl.setAttribute("x", 30); dl.setAttribute("y", 5);
  dl.setAttribute("fill", "rgba(255,255,255,0.5)");
  dl.setAttribute("font-size", 3);
  dl.setAttribute("text-anchor", "middle");
  dl.textContent = "defensive half — click here";
  svg.appendChild(dl);

  const markerGroup = document.createElementNS(svgNS, "g");
  markerGroup.setAttribute("id", "marker-group");
  svg.appendChild(markerGroup);

  svg.addEventListener("click", (ev) => {
    const pt = svg.createSVGPoint();
    pt.x = ev.clientX; pt.y = ev.clientY;
    const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
    let x = Math.max(0, Math.min(X_MAX_CLICK, loc.x));
    let y = Math.max(0, Math.min(PITCH_H, loc.y));
    setLocation(x, y);
  });

  pitchWrap.appendChild(svg);
}

function drawMarker(x, y) {
  const g = document.getElementById("marker-group");
  g.innerHTML = "";
  const halo = document.createElementNS(svgNS, "circle");
  halo.setAttribute("cx", x); halo.setAttribute("cy", y);
  halo.setAttribute("r", 3.5); halo.setAttribute("fill", "none");
  halo.setAttribute("stroke", "#ff4d5a"); halo.setAttribute("stroke-width", 0.3);
  halo.setAttribute("opacity", 0.7);
  g.appendChild(halo);
  const c = document.createElementNS(svgNS, "circle");
  c.setAttribute("cx", x); c.setAttribute("cy", y);
  c.setAttribute("r", 1.4); c.setAttribute("fill", "#ff4d5a");
  c.setAttribute("stroke", "white"); c.setAttribute("stroke-width", 0.3);
  g.appendChild(c);
}

// ===== Grid lookup =====
function nearestFlatXY(x, y) {
  const g = DATA.grid;
  const ix = Math.max(0, Math.min(g.nx - 1, Math.round((x - g.x_min) / (g.x_max - g.x_min) * (g.nx - 1))));
  const iy = Math.max(0, Math.min(g.ny - 1, Math.round((y - g.y_min) / (g.y_max - g.y_min) * (g.ny - 1))));
  return ix * g.ny + iy;
}

// ===== Helpers =====
function sign4(v) { return (v >= 0 ? "+" : "") + v.toFixed(4); }
function sign5(v) { return (v >= 0 ? "+" : "") + v.toFixed(5); }
function pctStr(v) { return (v * 100).toFixed(2) + "%"; }
function fmt(v, d = 4) { return (v >= 0 ? "+" : "") + v.toFixed(d); }

function approxPercentile(v) {
  const qs = DATA.percentiles;
  for (let p = 0; p <= 100; p++) if (v <= qs[p]) return p;
  return 100;
}

// ===== SOP sliders =====
function buildSopSliders(containerId, stateKey, onChange) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  DATA.sop_cols.forEach((col, idx) => {
    const positions = DATA.sop_positions[col];  // length SN
    const row = document.createElement("div");
    row.className = "slider-row";
    const lbl = document.createElement("div");
    lbl.className = "lbl"; lbl.textContent = SOP_LABELS[col] || col;
    lbl.title = col;  // show raw feature name on hover
    row.appendChild(lbl);
    const input = document.createElement("input");
    input.type = "range";
    input.min = 0; input.max = positions.length - 1; input.step = 1;
    input.value = state[stateKey][idx];
    input.addEventListener("input", () => {
      state[stateKey][idx] = parseInt(input.value);
      valEl.textContent = positions[state[stateKey][idx]].toFixed(3);
      onChange();
    });
    row.appendChild(input);
    const valEl = document.createElement("div");
    valEl.className = "val";
    valEl.textContent = positions[state[stateKey][idx]].toFixed(3);
    row.appendChild(valEl);
    const sub = document.createElement("div");
    sub.className = "sub";
    sub.style.gridColumn = "2";
    const ticks = positions.map(p => p.toFixed(2)).join(" · ");
    sub.textContent = `p05  p27  p50  p72  p95   (values: ${ticks})`;
    row.appendChild(sub);
    container.appendChild(row);
  });
}

// ===== Bars =====
function renderBars(containerId, values, labels, selectedIdx, opts) {
  opts = opts || {};
  const el = document.getElementById(containerId);
  el.innerHTML = "";
  const max = opts.maxOverride != null ? opts.maxOverride : Math.max(...values, 0.0001);
  values.forEach((v, i) => {
    const row = document.createElement("div");
    row.className = "bar-row" + (i === selectedIdx ? " selected" : "");
    const lbl = document.createElement("div"); lbl.className = "label"; lbl.textContent = labels[i];
    row.appendChild(lbl);
    const track = document.createElement("div"); track.className = "bar-track";
    const fill = document.createElement("div"); fill.className = "bar-fill";
    fill.style.width = `${Math.max(0, Math.min(100, (v / max) * 100))}%`;
    fill.style.background = CLS_COLOR[labels[i]] || "#8ac";
    track.appendChild(fill);
    row.appendChild(track);
    const val = document.createElement("div"); val.className = "value";
    val.textContent = opts.valueFmt ? opts.valueFmt(v) : pctStr(v);
    row.appendChild(val);
    el.appendChild(row);
  });
}

// ===== Formula =====
function renderFormula(p_out, p_goal, outcome_idx, expected, realized, ddv) {
  const classes = DATA.class_names;
  const observed = classes[outcome_idx];
  const terms = classes.map((c, i) =>
    `<span class="sym">p(${c})</span>·<span class="sym">p(goal|${c})</span>` +
    ` = <span class="val">${p_out[i].toFixed(4)}·${p_goal[i].toFixed(5)}</span>` +
    ` = <span class="val">${(p_out[i] * p_goal[i]).toFixed(6)}</span>`
  ).join("<br>");
  const html = `
    <div><b>expected danger</b> = Σ<sub>c</sub> p(c|x,y,speed of play) · p(goal|c,x,y,speed of play)</div>
    <div style="margin-left:14px">${terms}</div>
    <div style="margin-top:6px;margin-left:14px">sum = <span class="val">${expected.toFixed(6)}</span> = <span class="val">${pctStr(expected)}</span></div>
    <br>
    <div><b>realized danger</b> = p(goal | observed=<span style="color:${CLS_COLOR[observed]}">${observed}</span>, x, y, speed of play) = <span class="val">${realized.toFixed(6)}</span> = <span class="val">${pctStr(realized)}</span></div>
    <br>
    <div><b>DDV</b> = expected − realized = <span class="val">${expected.toFixed(6)}</span> − <span class="val">${realized.toFixed(6)}</span> = <span class="${ddv >= 0 ? 'pos' : 'neg'} val">${sign4(ddv)}</span></div>
  `;
  document.getElementById("formula-block").innerHTML = html;
  const box = document.getElementById("final-value");
  box.textContent = sign4(ddv);
  box.style.background = ddv >= 0 ? "rgba(110,231,183,0.15)" : "rgba(248,113,113,0.15)";
  box.style.color = ddv >= 0 ? "var(--green)" : "var(--red)";
  document.getElementById("final-explain").textContent =
    ddv >= 0 ? "defender averted more danger than expected" : "defender gave up more danger than expected";
}

// ===== Histogram marker =====
function renderHistogramMarker(ddv) {
  const { hist_lo, hist_hi, mean, std, n } = DATA.population;
  const img = document.getElementById("hist-img");
  const wrap = img.parentElement;
  const m = document.getElementById("hist-marker");
  const cap = document.getElementById("hist-marker-cap");
  document.getElementById("pct-mean").textContent = fmt(mean, 5);
  document.getElementById("pct-std").textContent = std.toFixed(5);
  document.getElementById("total-n").textContent = n.toLocaleString();
  if (ddv == null) {
    m.style.display = "none"; cap.style.display = "none";
    document.getElementById("pct-val").textContent = "—";
    document.getElementById("pct-pct").textContent = "—";
    return;
  }
  const percentile = approxPercentile(ddv);
  document.getElementById("pct-val").textContent = sign4(ddv);
  document.getElementById("pct-pct").textContent = `${percentile}th`;
  const LEFT_FRAC = DATA.population.hist_left_frac;
  const RIGHT_FRAC = DATA.population.hist_right_frac;
  const clamped = Math.max(hist_lo, Math.min(hist_hi, ddv));
  const t = (clamped - hist_lo) / (hist_hi - hist_lo);
  const frac = LEFT_FRAC + t * (RIGHT_FRAC - LEFT_FRAC);
  const rect = img.getBoundingClientRect();
  const wrapRect = wrap.getBoundingClientRect();
  const offset = rect.left - wrapRect.left;
  const left = offset + frac * rect.width;
  m.style.left = `${left}px`; m.style.display = "block";
  cap.style.left = `${left}px`; cap.style.display = "block";
  cap.textContent = `p${percentile} · ${sign4(ddv)}`;
}

// ===== Surface rendering (canvas) =====
//
// Each panel shows a (0..60) x (0..80) half-pitch with the defensive-half
// portion filled by a heatmap. We render at NX x NY cells upscaled to canvas
// pixels via ImageData, then overlay pitch lines.

// Canvas intrinsic size chosen to match the half-pitch aspect 60:80 (3:4) so the
// drawn pitch lines are isotropic and the surface isn't squished vertically.
const PANEL_W_PX = 240;   // internal canvas width in px
const PANEL_H_PX = 320;   // internal canvas height in px
const PITCH_SHOWN_X = 60; // show only defensive half to match trained region

function drawPitchOnCanvas(ctx, w, h) {
  ctx.save();
  // Coordinate mapping: pitch (0..60, 0..80) -> canvas (0..w, 0..h)
  const sx = w / PITCH_SHOWN_X;
  const sy = h / PITCH_H;
  ctx.strokeStyle = "rgba(255,255,255,0.55)";
  ctx.lineWidth = 1.1;
  ctx.lineJoin = "miter";
  const line = (x1,y1,x2,y2) => {
    ctx.beginPath();
    ctx.moveTo(x1*sx, y1*sy); ctx.lineTo(x2*sx, y2*sy); ctx.stroke();
  };
  const rect = (x,y,rw,rh) => { ctx.strokeRect(x*sx, y*sy, rw*sx, rh*sy); };
  const circ = (cx,cy,r) => {
    ctx.beginPath();
    ctx.ellipse(cx*sx, cy*sy, r*sx, r*sy, 0, 0, Math.PI*2);
    ctx.stroke();
  };
  // Outer + center-circle arc (only leftmost portion visible)
  rect(0,0,60,80);                // border
  line(60,0,60,80);               // halfway line (right edge of visible region)
  ctx.save(); ctx.beginPath();
  ctx.ellipse(60*sx, 40*sy, 9.15*sx, 9.15*sy, 0, Math.PI/2, 3*Math.PI/2);
  ctx.stroke(); ctx.restore();
  rect(0,18,18,44);               // penalty box
  rect(0,30,6,20);                // 6-yard box
  // Penalty spot + center spot
  ctx.fillStyle = "rgba(255,255,255,0.55)";
  ctx.beginPath(); ctx.arc(12*sx, 40*sy, 1.3, 0, Math.PI*2); ctx.fill();
  ctx.restore();
}

function drawHeatmap(canvas, surface /*Float32Array length G*/, vmin, vmax, cmapFn) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  const nx = DATA.grid.nx, ny = DATA.grid.ny;

  // Build small ImageData then scale with ctx.drawImage for pcolormesh feel.
  // Map surface cell (ix, iy) -> pixel at center of its region.
  // The surface grid covers x in [0, 60], y in [0, 80].
  // We'll render to an offscreen canvas at (nx, ny) then scale up.
  const off = document.createElement("canvas");
  off.width = nx; off.height = ny;
  const offCtx = off.getContext("2d");
  const img = offCtx.createImageData(nx, ny);
  const range = (vmax - vmin) || 1;
  for (let ix = 0; ix < nx; ix++) {
    for (let iy = 0; iy < ny; iy++) {
      const flat_xy = ix * ny + iy;
      const v = surface[flat_xy];
      const t = (v - vmin) / range;
      const rgb = cmapFn(t);
      // Canvas pixel (px, py): x axis is ix, y axis is iy. But pitch y=0 is TOP.
      // Our data follows statsbomb convention: y=0 top. So iy=0 -> top row.
      const px = ix;
      const py = iy;
      const off_idx = (py * nx + px) * 4;
      img.data[off_idx]     = rgb[0];
      img.data[off_idx + 1] = rgb[1];
      img.data[off_idx + 2] = rgb[2];
      img.data[off_idx + 3] = 230;  // alpha
    }
  }
  offCtx.putImageData(img, 0, 0);

  ctx.clearRect(0, 0, w, h);
  // Fill background pitch green so any uncovered margin looks right
  ctx.fillStyle = "#1b2a1f";
  ctx.fillRect(0, 0, w, h);

  // Scale the heatmap up with nearest-neighbor (no smoothing for clarity).
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, nx, ny, 0, 0, w, h);
  ctx.imageSmoothingEnabled = true;

  drawPitchOnCanvas(ctx, w, h);
}

function buildSurfaceRow() {
  const row = document.getElementById("surface-row");
  row.innerHTML = "";
  DATA.class_names.forEach((cls, i) => {
    const panel = document.createElement("div");
    panel.className = "surface-panel";
    const title = document.createElement("div");
    title.className = "surface-title";
    title.textContent = cls;
    panel.appendChild(title);
    const canvas = document.createElement("canvas");
    canvas.width = PANEL_W_PX; canvas.height = PANEL_H_PX;
    canvas.dataset.cls = cls;
    canvas.dataset.idx = i;
    panel.appendChild(canvas);
    const cbar = document.createElement("div");
    cbar.className = "colorbar";
    cbar.innerHTML = `<span class="lo">lo</span><div class="bar"></div><span class="hi">hi</span>`;
    panel.appendChild(cbar);
    row.appendChild(panel);
  });
}

function cmapForTab(tab) {
  if (tab === "outcome") return viridis;
  if (tab === "goal")    return magma;
  return diverging;   // ddv and ddv_duel
}

function colorbarGradient(cmapFn, n = 16) {
  const stops = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const c = cmapFn(t);
    stops.push(`rgb(${c[0]},${c[1]},${c[2]}) ${Math.round(t*100)}%`);
  }
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

function renderSurfaces() {
  const tab = state.surfaceTab;
  const [i, j, k] = state.sopSurface;
  const combo = comboIdx(i, j, k);
  const panels = document.querySelectorAll("#surface-row .surface-panel");

  // Pre-compute surfaces and shared color range per tab.
  const isDdv = (tab === "ddv" || tab === "ddv_duel");
  const surfaces = [];
  for (let c = 0; c < K; c++) {
    if (tab === "outcome") {
      surfaces.push(sliceSurface(P_OUTCOME, combo, c));
    } else if (tab === "goal") {
      surfaces.push(sliceSurface(P_GOAL, combo, c));
    } else {
      // DDV surface per observed outcome c:
      // expected (same for all panels) minus p_goal[c]
      const goalC = sliceSurface(P_GOAL, combo, c);
      // expected = Σ_k  p_outcome[k] * p_goal[k]
      const exp = new Float32Array(G_SIZE);
      for (let kk = 0; kk < K; kk++) {
        const po = sliceSurface(P_OUTCOME, combo, kk);
        const pg = sliceSurface(P_GOAL, combo, kk);
        for (let f = 0; f < G_SIZE; f++) exp[f] += po[f] * pg[f];
      }
      const ddv = new Float32Array(G_SIZE);
      if (tab === "ddv") {
        for (let f = 0; f < G_SIZE; f++) ddv[f] = exp[f] - goalC[f];
      } else {
        // DDV × P(duel | x, y)
        for (let f = 0; f < G_SIZE; f++) ddv[f] = (exp[f] - goalC[f]) * P_DUEL[f];
      }
      surfaces.push(ddv);
    }
  }

  let vmin, vmax;
  if (isDdv) {
    let absmax = 0;
    for (const s of surfaces) for (let f = 0; f < s.length; f++) {
      const a = Math.abs(s[f]); if (a > absmax) absmax = a;
    }
    vmin = -absmax; vmax = absmax;
  } else {
    vmin = Infinity; vmax = -Infinity;
    for (const s of surfaces) for (let f = 0; f < s.length; f++) {
      if (s[f] < vmin) vmin = s[f];
      if (s[f] > vmax) vmax = s[f];
    }
  }

  const cmap = cmapForTab(tab);
  const cmapEff = (t) => {
    if (isDdv) return diverging(2 * t - 1);
    return cmap(t);
  };

  const ddvFmt = (v) => (tab === "ddv_duel" ? sign5(v) : sign4(v));
  panels.forEach((panel, idx) => {
    const cls = DATA.class_names[idx];
    const canvas = panel.querySelector("canvas");
    drawHeatmap(canvas, surfaces[idx], vmin, vmax, cmapEff);
    const title = panel.querySelector(".surface-title");
    if (tab === "outcome")  title.innerHTML = `<span class="badge ${cls}">${cls}</span>  P(${cls} | x, y, speed&nbsp;of&nbsp;play)`;
    if (tab === "goal")     title.innerHTML = `<span class="badge ${cls}">${cls}</span>  P(goal | x, y, ${cls}, speed&nbsp;of&nbsp;play)`;
    if (tab === "ddv")      title.innerHTML = `<span class="badge ${cls}">${cls}</span>  DDV if outcome=${cls}`;
    if (tab === "ddv_duel") title.innerHTML = `<span class="badge ${cls}">${cls}</span>  DDV × P(duel) if outcome=${cls}`;
    const cbar = panel.querySelector(".colorbar .bar");
    cbar.style.background = colorbarGradient(cmapEff);
    panel.querySelector(".colorbar .lo").textContent = isDdv ? ddvFmt(vmin) : (vmin).toFixed(4);
    panel.querySelector(".colorbar .hi").textContent = isDdv ? ddvFmt(vmax) : (vmax).toFixed(4);
  });
}

// ===== Main render (top section) =====
function setLocation(x, y) {
  state.x = x; state.y = y;
  drawMarker(x, y);
  document.getElementById("meta-x").textContent = x.toFixed(1);
  document.getElementById("meta-y").textContent = y.toFixed(1);
  renderBreakdown();
}
function setOutcome(name) {
  state.outcome = name;
  document.querySelectorAll(".outcome-btn").forEach(b => {
    b.classList.toggle("selected", b.dataset.outcome === name);
  });
  document.getElementById("meta-outcome").innerHTML = `<span class="badge ${name}">${name}</span>`;
  renderBreakdown();
}

function renderBreakdown() {
  const empty = document.getElementById("empty-state");
  const breakdown = document.getElementById("breakdown");
  if (state.x == null || state.outcome == null) {
    empty.style.display = "block";
    breakdown.style.display = "none";
    renderHistogramMarker(null);
    return;
  }
  empty.style.display = "none";
  breakdown.style.display = "block";

  const flat = nearestFlatXY(state.x, state.y);
  const [i, j, k] = state.sopMain;
  const combo = comboIdx(i, j, k);
  const p_out = readOutcomeVec(combo, flat);
  const p_goal = readGoalVec(combo, flat);
  const classes = DATA.class_names;
  const outcome_idx = classes.indexOf(state.outcome);
  const expected = p_out.reduce((s, p, ii) => s + p * p_goal[ii], 0);
  const realized = p_goal[outcome_idx];
  const ddv = expected - realized;

  renderBars("bars-outcome", p_out, classes, outcome_idx, { maxOverride: 1.0 });
  renderBars("bars-goal", p_goal, classes, outcome_idx, {
    maxOverride: DATA.grid.p_goal_global_max,
    valueFmt: pctStr,
  });
  renderFormula(p_out, p_goal, outcome_idx, expected, realized, ddv);
  renderHistogramMarker(ddv);
}

// ===== Wiring =====
document.querySelectorAll(".outcome-btn").forEach(b => {
  b.addEventListener("click", () => setOutcome(b.dataset.outcome));
});
document.querySelectorAll(".tab").forEach(t => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    t.classList.add("active");
    state.surfaceTab = t.dataset.tab;
    renderSurfaces();
  });
});

// ===== Init =====
buildPitch();
buildSopSliders("sop-sliders-main", "sopMain", renderBreakdown);
buildSopSliders("sop-sliders-surface", "sopSurface", renderSurfaces);
buildSurfaceRow();

document.getElementById("hist-img").src = DATA.images.histogram;
document.getElementById("hist-img").addEventListener("load", () => {
  if (state.x != null && state.outcome != null) renderBreakdown();
});

window.addEventListener("resize", () => {
  if (state.x != null && state.outcome != null) renderBreakdown();
});

renderBreakdown();
renderSurfaces();
"""


def main():
    payload = build_payload()
    js = JS_TEMPLATE.replace(
        "/*__DATA__*/ {}",
        json.dumps(payload, separators=(",", ":")),
    )
    OUT_JS.write_text(js, encoding="utf-8")
    OUT_HTML.write_text(HTML_TEMPLATE, encoding="utf-8")
    print(f"Wrote {OUT_HTML}  ({OUT_HTML.stat().st_size / 1024:.1f} KB)")
    print(f"Wrote {OUT_JS}    ({OUT_JS.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
