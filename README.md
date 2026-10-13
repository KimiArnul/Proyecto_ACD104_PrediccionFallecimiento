# Predicción de supervivencia en pacientes con cirrosis: riesgos competitivos

Proyecto académico para la matería de **Aprendizaje Automático y Ciencia de Datos - ACD104**. El análisis utiliza el conjunto **Cirrhosis Patient Survival Prediction** del UC Irvine Machine Learning Repository (UCI, ID 878).

## Objetivo

Estimar y comparar la incidencia acumulada de fallecimiento antes del trasplante hepático, distinguiendo el fallecimiento (evento de interés), el trasplante (evento competitivo) y el final del seguimiento sin fallecimiento observado (censura).

> **Alcance y cautela:** el conjunto no contiene la supervivencia contrafactual de los pacientes trasplantados ni su seguimiento posterior. Por tanto, el proyecto no imputa cuánto habrían vivido sin trasplante y no debe utilizarse para tomar decisiones clínicas.

## Tres modelos predictivos

1. **Cox de causa específica:** analiza asociaciones con la tasa instantánea de cada evento. Para obtener una incidencia acumulada de fallecimiento, el código ajusta un Cox para fallecimiento y otro para trasplante y combina sus riesgos de causa específica.
2. **Fine-Gray:** modela la subdistribución del fallecimiento y permite estimar su incidencia acumulada en presencia del trasplante como evento competitivo.
3. **Bosque aleatorio de supervivencia con riesgos competitivos:** modelo basado en árboles que aprende relaciones potencialmente no lineales y estima curvas de incidencia acumulada.

El estimador no paramétrico de **Aalen-Johansen** se usa como referencia descriptiva de la incidencia acumulada observada; no cuenta como uno de los tres modelos predictivos.

## Estructura

```text
Proyecto_Cirrosis_Supervivencia/
├── README.md
├── requirements.txt
├── docs/
│   ├── 'Informe Proyecto de Catedra - ACD104.pdf'
│   └── 'Presentación Supervivencia Cirrosis.pdf'
├── notebooks/
│   ├── 01_auditoria_y_EDA.ipynb
│   └── 02_modelado_riesgos_competitivos.ipynb
├── src/
│   ├── cirrhosis_data.py
│   └── survival_models.py
├── data/raw/       # el CSV original se descargará aquí en la primera ejecución
└── results/
    ├── tables/     # métricas, coeficientes, predicciones OOF y auditoría
    └── figures/    # gráficos generados automáticamente
```

## Entorno recomendado

- Python **3.11 o 3.12 o 3.13**.
- Jupyter Notebook/JupyterLab.
- La primera descarga necesita conexión a internet.
- `comprisk==0.8.0` se fija para que el entorno sea reproducible. La documentación del paquete marca la API como *alpha*, por lo que se registra como limitación técnica y deberá mantenerse la versión fijada.

## Instalación en Windows (PowerShell)

Desde la carpeta `Proyecto_Cirrosis_Supervivencia`:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name cirrosis-supervivencia --display-name "Python (Cirrosis Supervivencia)"
jupyter notebook
```

Si no dispone de Python 3.13, instale Python 3.12 o 3.11, o adapte `py -3.12` a la versión disponible. En algunas configuraciones de PowerShell las políticas locales podrían bloquear la activación; como alternativa, ejecute `\.venv\Scripts\python.exe -m pip install -r requirements.txt` y `\.venv\Scripts\python.exe -m jupyter notebook`.

## Orden de ejecución

1. Abrir y ejecutar **todas las celdas** de `notebooks/01_auditoria_y_EDA.ipynb` de arriba abajo. El notebook descarga la fuente de UCI si no encuentra el CSV local, genera tablas de auditoría y guarda las figuras exploratorias.
2. Ejecutar `notebooks/02_modelado_riesgos_competitivos.ipynb`. Ajusta los modelos mediante validación cruzada estratificada de cinco pliegues, genera predicciones fuera de pliegue (OOF) y calcula AUC dependiente del tiempo, Brier score, métricas integradas, concordancia, coeficientes y calibración.
3. Revisar los CSV en `results/tables/` y las figuras en `results/figures/`. El informe Word en `docs/` incluye espacios para incorporar estas salidas una vez que se hayan verificado.

## Fuente de datos

UCI Machine Learning Repository: [Cirrhosis Patient Survival Prediction](https://archive.ics.uci.edu/dataset/878/cirrhosis+patient+survival+prediction+dataset-1). DOI: <https://doi.org/10.24432/C5R02G>. La ficha consultada declara licencia CC BY 4.0; compruebe y conserve la atribución de la fuente. Si la descarga automatizada no funciona, descargue el CSV original desde UCI y guárdelo como `data/raw/cirrhosis_uci_original.csv`.

## Codificación de eventos

| Status original | Interpretación | Código usado |
|---|---|---:|
| `C` | Censura: no se observó el fallecimiento durante el seguimiento | 0 |
| `D` | Fallecimiento | 1 |
| `CL` | Trasplante hepático antes del cierre del seguimiento | 2 |

El trasplante se modela como **evento competitivo**, no como una muerte ni como supervivencia indefinida. El supuesto de que los pacientes trasplantados habrían vivido menos tiempo sin el procedimiento no se trata como un hecho observado; el conjunto no permite identificar directamente ese contrafactual.

## Predictores usados para los modelos

El análisis exploratorio revisa las variables disponibles; por parsimonia y por su fundamento histórico en el modelo pronóstico de Mayo para PBC, la comparación de los tres modelos utiliza cinco predictores: edad, bilirrubina, albúmina, tiempo de protrombina y edema. Se generan `Age_years`, `log_Bilirubin`, `log_Albumin`, `log_Prothrombin` y `Edema_score` (N=0, S=0.5, Y=1). El preprocesamiento se ajusta dentro de cada pliegue para evitar fuga de información.

## Resultados generados

- Auditoría de calidad y valores faltantes.
- Frecuencia de `Status` y estadísticos descriptivos.
- Figuras exploratorias.
- Curvas estimadas de incidencia acumulada de fallecimiento y trasplante.
- Predicciones OOF por cada horizonte.
- AUC(t), Brier(t), iAUC, IBS y concordancia por causa.
- Tabla de calibración, coeficientes por pliegue y diagnóstico de convergencia de Cox y Fine-Gray.
- Importancia de variables del bosque ajustado sobre todos los datos, separada de la evaluación OOF.

**Importante sobre las métricas:** los intervalos bootstrap de `score_cr` se calculan sobre las predicciones OOF existentes; no equivalen a un bootstrap que reentrene por completo todos los modelos. Los tres años (365, 730 y 1,096 días aproximados) son horizontes de evaluación predefinidos, y deben interpretarse solo si la ejecución confirma soporte temporal suficiente.

## Limitaciones

- Muestra histórica y relativamente pequeña (UCI declara 418 observaciones).
- Solo se observa el seguimiento previo al trasplante, no el resultado posterior.
- La censura puede ser informativa y la indicación del trasplante puede depender de la gravedad clínica.
- No existe validación externa contemporánea; no es una herramienta para diagnóstico ni pronóstico individual.
- `comprisk` 0.8.0 estaba clasificado como paquete *alpha* en la fecha de preparación de este proyecto; mantenga la versión fijada y comunique cualquier problema de API en lugar de cambiar de versión silenciosamente.

## Si algo falla

- Verifique que el kernel de Jupyter sea `Python (Cirrosis Supervivencia)`.
- Ejecute `python -m pip show comprisk` y confirme la versión 0.8.0.
- Confirme que `data/raw/cirrhosis_uci_original.csv` contiene `N_Days`, `Status`, `Age`, `Bilirubin`, `Albumin`, `Prothrombin` y `Edema`.
- Si hay errores numéricos en el ajuste, conserve el traceback y los CSV de auditoría. No cambie el tratamiento de estados ni los horizontes sin documentarlo.

## Referencias técnicas principales

- UCI Machine Learning Repository. *Cirrhosis Patient Survival Prediction*.
- Dickson, E. R., et al. (1989). *Prognosis in primary biliary cirrhosis: Model for decision making*. Hepatology, 10(1), 1–7. <https://doi.org/10.1002/hep.1840100102>
- Yang, S., Zhao, W., & Zhao, W. (2026). *comprisk: A scikit-learn-compatible Python toolkit for competing-risks survival analysis*. arXiv:2607.09431. <https://arxiv.org/abs/2607.09431>
