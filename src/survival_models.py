"""Cross-validated competing-risks models and evaluation utilities.

Models: cause-specific Cox, Fine-Gray and competing-risks random survival forest.
Event coding: 0=censored, 1=death, 2=liver transplant (competing event).
"""
from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from .cirrhosis_data import clinical_predictors, survival_outcomes


def _cox_baseline_increments(model, X_train, time_train, event_train, cause: int):
    """Breslow baseline cumulative-hazard increments for one cause-specific Cox fit."""
    linear = np.asarray(model.predict(X_train), dtype=float).reshape(-1)
    relative_risk = np.exp(np.clip(linear, -30, 30))
    event_times = np.sort(np.unique(time_train[event_train == cause]))
    increments = {}
    for t in event_times:
        d = int(np.sum((time_train == t) & (event_train == cause)))
        at_risk = time_train >= t
        denom = float(relative_risk[at_risk].sum())
        if denom > 0 and d > 0:
            increments[float(t)] = d / denom
    return increments


def _cox_competing_risk_cif(
    cox_death, cox_transplant, X_train, time_train, event_train, X_eval, eval_times
):
    """Approximate death CIF by combining two cause-specific Cox hazards.

    Uses Breslow increments for each cause and integrates the combined cause-specific
    hazards over the union of observed event times. It produces a CIF matrix compatible
    with competing-risk scoring; document the approximation and inspect diagnostics.
    """
    inc_death = _cox_baseline_increments(cox_death, X_train, time_train, event_train, 1)
    inc_tx = _cox_baseline_increments(cox_transplant, X_train, time_train, event_train, 2)
    grid = np.array(sorted(set(inc_death) | set(inc_tx)), dtype=float)
    lp_death = np.exp(np.clip(np.asarray(cox_death.predict(X_eval), dtype=float).reshape(-1), -30, 30))
    lp_tx = np.exp(np.clip(np.asarray(cox_transplant.predict(X_eval), dtype=float).reshape(-1), -30, 30))
    prediction = np.zeros((len(X_eval), len(eval_times)), dtype=float)

    for j, horizon in enumerate(eval_times):
        surv = np.ones(len(X_eval), dtype=float)
        cif_death = np.zeros(len(X_eval), dtype=float)
        for t in grid[grid <= horizon]:
            dh1 = inc_death.get(float(t), 0.0) * lp_death
            dh2 = inc_tx.get(float(t), 0.0) * lp_tx
            dh_total = dh1 + dh2
            event_probability = -np.expm1(-dh_total)
            share_death = np.divide(dh1, dh_total, out=np.zeros_like(dh1), where=dh_total > 0)
            cif_death += surv * event_probability * share_death
            surv *= np.exp(-dh_total)
        prediction[:, j] = np.clip(cif_death, 0.0, 1.0)
    return prediction


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def run_competing_risk_analysis(
    df: pd.DataFrame,
    output_dir: str | Path = "results",
    n_splits: int = 5,
    eval_times=(365.25, 730.5, 1095.75),
    n_estimators: int = 300,
    n_bootstrap: int = 200,
    random_state: int = 42,
    n_jobs: int = 1,
) -> dict[str, pd.DataFrame]:
    """Run stratified out-of-fold comparison and export tables/figures.

    Predictions for metrics are out-of-fold: every patient's prediction is produced by a
    model that was fitted without that patient's outcome. Bootstrap intervals quantify
    test-sample uncertainty conditional on those out-of-fold predictions; they do not
    represent a full nested-bootstrap refitting analysis.
    """
    try:
        from comprisk import (
            CauseSpecificCox,
            FineGrayRegression,
            CompetingRiskForest,
            CumulativeIncidence,
            score_cr,
            concordance_index_ci,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar comprisk. Instale exactamente las dependencias del proyecto con "
            "`python -m pip install -r requirements.txt`. El paquete comprisk se fija en 0.8.0 "
            "porque su API aún se encuentra en etapa alpha."
        ) from exc

    out = Path(output_dir)
    tables_dir = out / "tables"
    figures_dir = out / "figures"
    models_dir = out / "models"
    for folder in (tables_dir, figures_dir, models_dir):
        folder.mkdir(parents=True, exist_ok=True)

    X_raw = clinical_predictors(df)
    time, event, labels = survival_outcomes(df)
    eval_times = np.asarray(eval_times, dtype=float)
    if eval_times.ndim != 1 or len(eval_times) == 0 or np.any(eval_times <= 0) or np.any(np.diff(eval_times) <= 0):
        raise ValueError("eval_times debe ser una lista creciente de tiempos positivos en días.")
    if np.max(eval_times) >= np.max(time):
        raise ValueError(
            "Al menos un horizonte de evaluación alcanza/supera la duración máxima observada; "
            "seleccione horizontes con soporte temporal suficiente y documente la decisión."
        )
    counts = pd.Series(labels).value_counts()
    if counts.min() < n_splits:
        raise ValueError(
            f"No hay suficientes observaciones en la clase minoritaria para {n_splits} pliegues. "
            f"Conteos: {counts.to_dict()}"
        )

    model_names = ["Cox causa-específica", "Fine-Gray", "Bosque supervivencia riesgos competitivos"]
    oof_cif = {name: np.full((len(df), len(eval_times)), np.nan) for name in model_names}
    oof_risk = {name: np.full(len(df), np.nan) for name in model_names}
    coef_records = []
    convergence_records = []
    fold_id = np.full(len(df), -1, dtype=int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_raw, labels), start=1):
        xtr_raw = X_raw.iloc[train_idx]
        xte_raw = X_raw.iloc[test_idx]
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        scaler = StandardScaler()
        xtr_imp = imputer.fit_transform(xtr_raw)
        xte_imp = imputer.transform(xte_raw)
        xtr = scaler.fit_transform(xtr_imp)
        xte = scaler.transform(xte_imp)

        ttr, etr = time[train_idx], event[train_idx]
        tte, ete = time[test_idx], event[test_idx]

        cox_death = CauseSpecificCox(cause=1, max_iter=100).fit(xtr, time=ttr, event=etr)
        cox_tx = CauseSpecificCox(cause=2, max_iter=100).fit(xtr, time=ttr, event=etr)
        finegray = FineGrayRegression(cause=1, robust_se=True).fit(xtr, time=ttr, event=etr)
        forest = CompetingRiskForest(
            n_estimators=n_estimators,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=random_state + fold,
            n_jobs=n_jobs,
            splitrule="logrankCR",
        ).fit(xtr, ttr, etr)

        cox_cif = _cox_competing_risk_cif(cox_death, cox_tx, xtr, ttr, etr, xte, eval_times)
        fg_cif = np.asarray(finegray.predict_cumulative_incidence(xte, times=eval_times), dtype=float)
        forest_cif_all = np.asarray(forest.predict_cif(xte, times=eval_times), dtype=float)
        if forest_cif_all.ndim != 3 or forest_cif_all.shape[1] < 1:
            raise RuntimeError(f"La forma de CIF devuelta por el bosque no es la esperada: {forest_cif_all.shape}")
        forest_cif = forest_cif_all[:, 0, :]  # cause code 1 = death

        predictions = {
            model_names[0]: cox_cif,
            model_names[1]: fg_cif,
            model_names[2]: forest_cif,
        }
        for name, matrix in predictions.items():
            if matrix.shape != (len(test_idx), len(eval_times)):
                raise RuntimeError(f"Predicciones CIF inesperadas para {name}: {matrix.shape}")
            if not np.isfinite(matrix).all():
                raise RuntimeError(f"El modelo {name} produjo probabilidades no finitas.")
            if np.any((matrix < -1e-8) | (matrix > 1 + 1e-8)):
                raise RuntimeError(f"El modelo {name} produjo probabilidades fuera de [0,1].")
            oof_cif[name][test_idx, :] = np.clip(matrix, 0, 1)

        risk_vectors = {
            model_names[0]: np.asarray(cox_death.predict(xte)).reshape(-1),
            model_names[1]: np.asarray(finegray.predict(xte)).reshape(-1),
            model_names[2]: np.asarray(forest.predict_risk(xte, cause=1)).reshape(-1),
        }
        for name, risk in risk_vectors.items():
            oof_risk[name][test_idx] = risk

        for model_name, model, feature_names, effect_name in [
            ("Cox causa-específica: fallecimiento", cox_death, X_raw.columns, "HR causa-específica"),
            ("Cox causa-específica: trasplante", cox_tx, X_raw.columns, "HR causa-específica"),
            ("Fine-Gray: fallecimiento", finegray, X_raw.columns, "SHR de subdistribución"),
        ]:
            coefs = np.asarray(model.coef_, dtype=float).reshape(-1)
            for feature, beta in zip(feature_names, coefs):
                coef_records.append({
                    "pliegue": fold,
                    "modelo": model_name,
                    "variable": feature,
                    "coeficiente_escala_estandarizada": float(beta),
                    "razon_riesgo_exponencial": float(np.exp(np.clip(beta, -30, 30))),
                    "tipo_efecto": effect_name,
                })
        convergence_records.extend([
            {"pliegue": fold, "modelo": "Cox causa-específica: fallecimiento", "convergio": bool(getattr(cox_death, "converged_", True)), "iteraciones": getattr(cox_death, "n_iter_", np.nan)},
            {"pliegue": fold, "modelo": "Cox causa-específica: trasplante", "convergio": bool(getattr(cox_tx, "converged_", True)), "iteraciones": getattr(cox_tx, "n_iter_", np.nan)},
            {"pliegue": fold, "modelo": "Fine-Gray: fallecimiento", "convergio": bool(getattr(finegray, "converged_", True)), "iteraciones": getattr(finegray, "n_iter_", np.nan)},
        ])
        fold_id[test_idx] = fold
        print(f"Pliegue {fold}/{n_splits} finalizado.")

    for name, matrix in oof_cif.items():
        if np.isnan(matrix).any():
            raise RuntimeError(f"Faltan predicciones fuera de muestra para {name}.")
    for name, vector in oof_risk.items():
        if np.isnan(vector).any():
            raise RuntimeError(f"Faltan puntajes de riesgo fuera de muestra para {name}.")

    prediction_dict = {name: mat for name, mat in oof_cif.items()}
    score_result = score_cr(
        prediction_dict,
        test_time=time,
        test_event=event,
        eval_times=eval_times,
        cause=1,
        metrics=("auc", "brier"),
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        n_jobs=n_jobs,
        calibration_at=[float(eval_times[-1])],
        calibration_n_bins=5,
    )
    auc_df = score_result.auc.copy()
    brier_df = score_result.brier.copy()
    iauc_df = score_result.iauc.copy()
    ibs_df = score_result.ibs.copy()
    calibration_df = score_result.calibration.copy() if getattr(score_result, "calibration", None) is not None else pd.DataFrame()

    concordance_records = []
    for name, risk in oof_risk.items():
        ci = concordance_index_ci(event, time, risk, cause=1)
        concordance_records.append({
            "modelo": name,
            "c_index_uno": float(ci.estimate),
            "limite_inferior_95": float(ci.ci_low),
            "limite_superior_95": float(ci.ci_high),
            "nota": "Concordancia por causa en predicciones fuera de pliegue; complemento del AUC/Brier temporal.",
        })
    concordance_df = pd.DataFrame(concordance_records)

    coef_df = pd.DataFrame(coef_records)
    convergence_df = pd.DataFrame(convergence_records)
    oof_df = pd.DataFrame({"N_Days": time, "Status_label": labels, "event_code": event, "pliegue_oof": fold_id})
    for name, matrix in oof_cif.items():
        safe = name.lower().replace(" ", "_").replace(":", "").replace("-", "_")
        for col, horizon in enumerate(eval_times):
            oof_df[f"CIF_fallecimiento__{safe}__{int(round(horizon))}d"] = matrix[:, col]
    for name, risk in oof_risk.items():
        safe = name.lower().replace(" ", "_").replace(":", "").replace("-", "_")
        oof_df[f"riesgo__{safe}"] = risk

    # Descriptive Aalen-Johansen CIF over the full observed cohort.
    aj = CumulativeIncidence(cause_codes=[1, 2]).fit(time=time, event=event)
    grid = np.linspace(max(1.0, float(np.min(time))), min(float(np.quantile(time, 0.95)), float(np.max(time))), 100)
    aj_est, aj_var = aj.timepoints(grid)
    aj_records = []
    for cause_idx, cause_code in enumerate([1, 2]):
        for j, t in enumerate(grid):
            value = float(aj_est[cause_idx, j])
            variance = float(max(0.0, aj_var[cause_idx, j]))
            aj_records.append({
                "tiempo_dias": float(t),
                "causa": "Fallecimiento" if cause_code == 1 else "Trasplante",
                "incidencia_acumulada": value,
                "error_estandar": float(np.sqrt(variance)),
            })
    aj_df = pd.DataFrame(aj_records)

    # Full-data forest used only for exploratory variable importance, not for performance scoring.
    full_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    full_scaler = StandardScaler()
    X_imp = full_imputer.fit_transform(X_raw)
    X_all = full_scaler.fit_transform(X_imp)
    forest_full = CompetingRiskForest(
        n_estimators=n_estimators, max_depth=5, min_samples_split=10,
        min_samples_leaf=8, max_features="sqrt", random_state=random_state,
        n_jobs=n_jobs, splitrule="logrankCR"
    ).fit(X_all, time, event)
    try:
        importance_df = forest_full.compute_importance(random_state=random_state)
        if not isinstance(importance_df, pd.DataFrame):
            importance_df = pd.DataFrame(importance_df)
    except Exception as exc:
        warnings.warn(f"No se pudo calcular la importancia OOB del bosque: {exc}")
        importance_df = pd.DataFrame()

    _write_table(auc_df, tables_dir / "auc_dependiente_del_tiempo.csv")
    _write_table(brier_df, tables_dir / "brier_dependiente_del_tiempo.csv")
    _write_table(iauc_df, tables_dir / "auc_integrada.csv")
    _write_table(ibs_df, tables_dir / "brier_integrado.csv")
    _write_table(concordance_df, tables_dir / "concordancia_uno.csv")
    _write_table(coef_df, tables_dir / "coeficientes_por_pliegue.csv")
    _write_table(convergence_df, tables_dir / "diagnostico_convergencia_modelos.csv")
    _write_table(calibration_df, tables_dir / "calibracion_horizonte_final.csv")
    _write_table(oof_df, tables_dir / "predicciones_fuera_de_pliegue.csv")
    _write_table(aj_df, tables_dir / "incidencia_acumulada_aalen_johansen.csv")
    _write_table(importance_df, tables_dir / "importancia_variables_bosque_completo.csv")

    # Figures
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in [("Fallecimiento", "#B23A48"), ("Trasplante", "#2F6690")]:
        sub = aj_df[aj_df["causa"] == label]
        ax.plot(sub["tiempo_dias"] / 365.25, sub["incidencia_acumulada"], label=label, linewidth=2, color=color)
    ax.set(title="Incidencia acumulada observada por tipo de evento", xlabel="Años desde el registro", ylabel="Incidencia acumulada")
    ax.set_ylim(0, 1); ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(figures_dir / "01_incidencia_acumulada_aj.png", dpi=180); plt.close(fig)

    # Compare mean OOF CIF predictions with observed Aalen-Johansen estimates at declared horizons.
    aj_h, aj_v = aj.timepoints(eval_times)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eval_times / 365.25, aj_h[0, :], marker="o", linewidth=2, label="Aalen-Johansen observado", color="#222222")
    for name, mat in oof_cif.items():
        ax.plot(eval_times / 365.25, mat.mean(axis=0), marker="o", label=name)
    ax.set(title="Incidencia acumulada observada y media predicha", xlabel="Años desde el registro", ylabel="Incidencia acumulada de fallecimiento")
    ax.set_ylim(0, 1); ax.grid(alpha=0.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figures_dir / "02_cif_observada_vs_predicha.png", dpi=180); plt.close(fig)

    if not auc_df.empty and {"model", "times", "AUC"}.issubset(auc_df.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, sub in auc_df.groupby("model"):
            ax.plot(sub["times"] / 365.25, sub["AUC"], marker="o", label=name)
        ax.set(title="AUC dependiente del tiempo", xlabel="Años desde el registro", ylabel="AUC(t)")
        ax.set_ylim(0, 1); ax.grid(alpha=0.25); ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(figures_dir / "03_auc_dependiente_tiempo.png", dpi=180); plt.close(fig)
    if not brier_df.empty and {"model", "times", "Brier"}.issubset(brier_df.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, sub in brier_df.groupby("model"):
            ax.plot(sub["times"] / 365.25, sub["Brier"], marker="o", label=name)
        ax.set(title="Brier score dependiente del tiempo", xlabel="Años desde el registro", ylabel="Brier(t); menor es mejor")
        ax.grid(alpha=0.25); ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(figures_dir / "04_brier_dependiente_tiempo.png", dpi=180); plt.close(fig)

    if not calibration_df.empty:
        needed = {"model", "times", "predicted_decile", "observed_freq", "lower_ci", "upper_ci"}
        if needed.issubset(calibration_df.columns):
            cal = calibration_df.loc[
                np.isclose(pd.to_numeric(calibration_df["times"], errors="coerce"), float(eval_times[-1]))
            ].copy()
            if not cal.empty:
                fig, ax = plt.subplots(figsize=(7, 6))
                max_value = float(np.nanmax(cal[["predicted_decile", "observed_freq", "upper_ci"]].to_numpy()))
                max_value = min(1.0, max(0.05, max_value * 1.08))
                ax.plot([0, max_value], [0, max_value], linestyle="--", color="#666666", label="Calibración ideal")
                for name, sub in cal.groupby("model"):
                    sub = sub.sort_values("predicted_decile")
                    yerr = np.vstack([
                        np.maximum(0, sub["observed_freq"].to_numpy() - sub["lower_ci"].to_numpy()),
                        np.maximum(0, sub["upper_ci"].to_numpy() - sub["observed_freq"].to_numpy()),
                    ])
                    ax.errorbar(
                        sub["predicted_decile"], sub["observed_freq"], yerr=yerr,
                        marker="o", capsize=2, linewidth=1.4, label=str(name),
                    )
                ax.set(
                    title=f"Calibración de la incidencia acumulada de fallecimiento a {eval_times[-1] / 365.25:.0f} años",
                    xlabel="Incidencia acumulada predicha (promedio por grupo)",
                    ylabel="Incidencia acumulada observada por Aalen-Johansen",
                    xlim=(0, max_value), ylim=(0, max_value),
                )
                ax.grid(alpha=0.25); ax.legend(fontsize=8); fig.tight_layout()
                fig.savefig(figures_dir / "06_calibracion_cif_horizonte_final.png", dpi=180)
                plt.close(fig)

    if not importance_df.empty:
        # Pick the available composite VIMP column for charting.
        vimp_col = "composite_vimp" if "composite_vimp" in importance_df.columns else next((c for c in importance_df.columns if "vimp" in c.lower()), None)
        feat_col = "feature" if "feature" in importance_df.columns else importance_df.columns[0]
        if vimp_col:
            vis = importance_df.sort_values(vimp_col, ascending=True)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.barh(vis[feat_col].astype(str), vis[vimp_col])
            ax.set(title="Importancia exploratoria de variables del bosque", xlabel="Importancia por permutación", ylabel="Variable")
            ax.grid(axis="x", alpha=0.25); fig.tight_layout()
            fig.savefig(figures_dir / "05_importancia_variables_bosque.png", dpi=180); plt.close(fig)

    return {
        "auc": auc_df,
        "brier": brier_df,
        "iauc": iauc_df,
        "ibs": ibs_df,
        "concordance": concordance_df,
        "coefficients": coef_df,
        "convergence": convergence_df,
        "calibration": calibration_df,
        "oof_predictions": oof_df,
        "aalen_johansen": aj_df,
        "variable_importance": importance_df,
    }
