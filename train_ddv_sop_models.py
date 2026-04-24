"""Train SOP-aware DDV models.

Like train_ddv_models.py, but both models also consume three SOP context
features (delta_x_30s, dist_30s, dist_ratio_30v60). Artifacts are saved under
./ddv_sop_models/ and consumed by build_ddv_explainer_sop.py.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss, classification_report

ROOT = Path(__file__).parent
PARQUET = ROOT / "data" / "defensive_duels_preprocessed.parquet"
OUT_DIR = ROOT / "ddv_sop_models"
OUT_DIR.mkdir(exist_ok=True)

SOP_COLS = ["delta_x_30s", "dist_30s", "dist_ratio_30v60"]


def load_duels() -> pd.DataFrame:
    duels = pd.read_parquet(PARQUET)
    d = duels[duels.is_defensive_duel & (duels.start_x < 60)].copy()
    d["groundDuel_stoppedProgress"] = (
        d["groundDuel_stoppedProgress"] & ~d["groundDuel_recoveredPossession"]
    )
    recovered = d["groundDuel_recoveredPossession"].fillna(False).astype(bool)
    stopped = d["groundDuel_stoppedProgress"].fillna(False).astype(bool)
    d["duel_outcome"] = np.select(
        [recovered, stopped], ["recovered", "stopped"], default="beat"
    )
    d = d.dropna(subset=SOP_COLS).reset_index(drop=True)
    return d


def train_outcome_model(X, y, class_names):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(class_names),
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=500,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    proba = model.predict_proba(X_te)
    base = np.bincount(y_tr) / len(y_tr)
    print(f"  outcome  logloss={log_loss(y_te, proba):.4f}   "
          f"baseline={log_loss(y_te, np.tile(base, (len(y_te), 1))):.4f}")
    print(classification_report(y_te, model.predict(X_te), target_names=class_names))
    return model


def train_goal_cond_model(X_xy_sop, y_outcome, y_goal):
    # features: [x, y, *sop_cols, outcome_code] — outcome at index 5
    Xc = np.column_stack([X_xy_sop, y_outcome])
    Xc_tr, Xc_te, yg_tr, yg_te = train_test_split(
        Xc, y_goal, test_size=0.2, random_state=42, stratify=y_goal
    )
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=500,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        Xc_tr, yg_tr,
        eval_set=[(Xc_te, yg_te)],
        eval_metric="binary_logloss",
        categorical_feature=[5],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    proba = model.predict_proba(Xc_te)[:, 1]
    print(f"  goal_cond  ROC_AUC={roc_auc_score(yg_te, proba):.4f}   "
          f"logloss={log_loss(yg_te, proba):.5f}")
    return model


def compute_population_values(outcome_model, goal_cond_model, X_xy_sop, y, n_classes):
    p_outcomes = outcome_model.predict_proba(X_xy_sop)
    p_goal_per_outcome = np.column_stack([
        goal_cond_model.predict_proba(
            np.column_stack([X_xy_sop, np.full(len(X_xy_sop), i)])
        )[:, 1]
        for i in range(n_classes)
    ])
    expected_danger = (p_outcomes * p_goal_per_outcome).sum(axis=1)
    realized_danger = p_goal_per_outcome[np.arange(len(X_xy_sop)), y]
    return expected_danger - realized_danger


def main():
    print(f"Loading {PARQUET} …")
    d = load_duels()
    print(f"  {len(d):,} defensive duels (defensive half, non-null SOP)")

    le = LabelEncoder()
    y = le.fit_transform(d["duel_outcome"])
    class_names = list(le.classes_)
    X_xy_sop = d[["start_x", "start_y"] + SOP_COLS].to_numpy()
    y_goal = (d["outcome"] == -1).astype(int).to_numpy()
    print(f"  classes: {class_names}")
    print(f"  positive rate (goal in 20s): {y_goal.mean():.5f}")

    print("\nTraining outcome model (x, y, sop → outcome) …")
    outcome_model = train_outcome_model(X_xy_sop, y, class_names)

    print("\nTraining conditional goal model (x, y, sop, outcome → goal) …")
    goal_cond_model = train_goal_cond_model(X_xy_sop, y, y_goal)

    print("\nComputing population DDV distribution …")
    values = compute_population_values(
        outcome_model, goal_cond_model, X_xy_sop, y, len(class_names)
    )
    print(f"  DDV  mean={values.mean():+.4e}   std={values.std():.4e}   "
          f"min={values.min():+.4f}   max={values.max():+.4f}")

    sop_arr = d[SOP_COLS].to_numpy()
    sop_meta = {
        "cols": SOP_COLS,
        "quantiles": {
            col: np.quantile(sop_arr[:, i], [0.05, 0.275, 0.50, 0.725, 0.95]).tolist()
            for i, col in enumerate(SOP_COLS)
        },
        "stats": {
            col: {
                "min": float(sop_arr[:, i].min()),
                "max": float(sop_arr[:, i].max()),
                "mean": float(sop_arr[:, i].mean()),
                "std": float(sop_arr[:, i].std()),
                "p05": float(np.quantile(sop_arr[:, i], 0.05)),
                "p50": float(np.quantile(sop_arr[:, i], 0.5)),
                "p95": float(np.quantile(sop_arr[:, i], 0.95)),
            }
            for i, col in enumerate(SOP_COLS)
        },
    }

    joblib.dump(outcome_model, OUT_DIR / "outcome_model.joblib")
    joblib.dump(goal_cond_model, OUT_DIR / "goal_cond_model.joblib")
    joblib.dump(le, OUT_DIR / "label_encoder.joblib")
    joblib.dump(sop_meta, OUT_DIR / "sop_meta.joblib")
    np.save(OUT_DIR / "population_values.npy", values)
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
