"""Carga, valida y prepara los datos de Cirrhosis Patient Survival Prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "cirrhosis_uci_original.csv"

COLUMN_ALIASES = {
    "tryglicerides": "Triglycerides",
    "triglycerides": "Triglycerides",
    "n_days": "N_Days",
    "status": "Status",
    "age": "Age",
    "edema": "Edema",
    "bilirubin": "Bilirubin",
    "albumin": "Albumin",
    "prothrombin": "Prothrombin",
}

STATUS_TEXT_TO_EVENT = {"C": 0, "D": 1, "CL": 2}
STATUS_NUMERIC_TO_TEXT = {"0": "D", "0.0": "D", "1": "C", "1.0": "C", "2": "CL", "2.0": "CL"}


def _standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    renamed = {}
    for col in df.columns:
        clean = str(col).strip().replace(" ", "_")
        key = clean.lower()
        renamed[col] = COLUMN_ALIASES.get(key, clean)
    df = df.rename(columns=renamed)
    # Avoid duplicated columns if source aliases normalize to the same name.
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def normalize_status(values: pd.Series) -> pd.Series:
    """Return canonical labels D, C, CL; supports source labels and UCI codes."""
    s = values.astype("string").str.strip().str.upper()
    observed = set(s.dropna().unique().tolist())
    if observed.issubset({"D", "C", "CL"}):
        return s
    # UCI metadata codes: 0 = D, 1 = C, 2 = CL.
    mapped = s.map(STATUS_NUMERIC_TO_TEXT)
    if mapped.notna().sum() == s.notna().sum():
        return mapped
    unexpected = sorted(str(v) for v in observed - {"D", "C", "CL"})
    raise ValueError(
        "La columna Status contiene categorías no reconocidas. "
        f"Valores observados: {unexpected}. Revise el archivo de UCI antes de continuar."
    )


def load_cirrhosis_data(
    path: Optional[str | Path] = None,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    """Load local CSV if present; otherwise download dataset 878 using ucimlrepo.

    The original source is preserved as downloaded. A canonical Status_label helper
    column is added in memory for analyses; it is not written back over Status.
    """
    csv_path = Path(path) if path else DEFAULT_RAW_PATH
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif download_if_missing:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise ImportError(
                "No se encontró el CSV local y falta ucimlrepo. Instale las dependencias "
                "con `python -m pip install -r requirements.txt` o descargue el CSV desde "
                "https://archive.ics.uci.edu/dataset/878/cirrhosis+patient+survival+prediction+dataset-1 "
                "y guárdelo en data/raw/cirrhosis_uci_original.csv."
            ) from exc
        dataset = fetch_ucirepo(id=878)
        original = getattr(dataset.data, "original", None)
        if isinstance(original, pd.DataFrame) and not original.empty:
            df = original.copy()
        else:
            frames = []
            for attr in ("ids", "features", "targets"):
                item = getattr(dataset.data, attr, None)
                if isinstance(item, pd.DataFrame) and not item.empty:
                    frames.append(item.copy())
            if not frames:
                raise RuntimeError("ucimlrepo no devolvió una tabla reconocible para el conjunto UCI 878.")
            df = pd.concat(frames, axis=1)
        df = _standardize_columns(df)
        if not {"N_Days", "Status"}.issubset(df.columns):
            raise RuntimeError(
                "La respuesta de UCI no contiene N_Days y Status. Descargue el CSV original "
                "desde la página del conjunto y colóquelo en data/raw/cirrhosis_uci_original.csv. "
                f"Columnas recibidas: {list(df.columns)}"
            )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    else:
        raise FileNotFoundError(f"No se encuentra el archivo de datos: {csv_path}")

    df = _standardize_columns(df)
    required = {"N_Days", "Status", "Age", "Bilirubin", "Albumin", "Prothrombin", "Edema"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing)}. "
            f"Columnas disponibles: {list(df.columns)}. Revise que se haya descargado el CSV original completo."
        )

    df["Status_label"] = normalize_status(df["Status"])
    df["N_Days"] = pd.to_numeric(df["N_Days"], errors="coerce")
    bad_outcome = df["N_Days"].isna() | df["Status_label"].isna()
    if bad_outcome.any():
        raise ValueError(
            f"Hay {int(bad_outcome.sum())} registros sin tiempo o estado verificable. "
            "No se eliminarán silenciosamente: revise el archivo original y documente la decisión."
        )
    if (df["N_Days"] < 0).any():
        raise ValueError("N_Days contiene tiempos negativos; revise la integridad del archivo fuente.")
    return df


def survival_outcomes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time (days), event codes (0=censored, 1=death, 2=transplant), labels."""
    labels = df["Status_label"] if "Status_label" in df else normalize_status(df["Status"])
    event = labels.map(STATUS_TEXT_TO_EVENT)
    if event.isna().any():
        raise ValueError("Se encontraron estados sin correspondencia con el esquema de eventos.")
    return (
        pd.to_numeric(df["N_Days"], errors="raise").to_numpy(dtype=float),
        event.to_numpy(dtype=int),
        labels.astype(str).to_numpy(),
    )


def clinical_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """Create a parsimonious, clinically motivated feature set based on the Mayo PBC model.

    Candidate predictors: age (years), log bilirubin, log albumin, log prothrombin time,
    and edema score N=0, S=0.5, Y=1. The source's remaining variables are still explored
    in the EDA notebook but are not automatically added to the small-event regression models.
    """
    x = pd.DataFrame(index=df.index)
    x["Age_years"] = pd.to_numeric(df["Age"], errors="coerce") / 365.25
    for source, output in [
        ("Bilirubin", "log_Bilirubin"),
        ("Albumin", "log_Albumin"),
        ("Prothrombin", "log_Prothrombin"),
    ]:
        values = pd.to_numeric(df[source], errors="coerce")
        values = values.where(values > 0, np.nan)
        x[output] = np.log(values)

    edema = df["Edema"].astype("string").str.strip().str.upper()
    edema_map = {
        "N": 0.0, "S": 0.5, "Y": 1.0,
        "0": 0.0, "0.0": 0.0, "0.5": 0.5, "1": 1.0, "1.0": 1.0,
    }
    x["Edema_score"] = edema.map(edema_map)
    x = x.replace([np.inf, -np.inf], np.nan)
    return x
