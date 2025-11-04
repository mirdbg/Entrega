# 🧠 MCPort — Monte Carlo Portfolio Toolkit

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mirdbg/Entrega1_MCPort/blob/main/notebooks/01_Quickstart.ipynb)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest%20passing-brightgreen.svg)](#-tests)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

> **MCPort** es un toolkit educativo y financiero para simular, analizar y visualizar carteras de inversión mediante métodos de **Monte Carlo**.  
> Incluye módulos para extracción de precios, cálculo de métricas, generación de informes PDF y visualizaciones interactivas.

---

## 🚀 Instalación local

```bash
# 1. Clonar el repositorio
git clone https://github.com/mirdbg/Entrega1_MCPort.git
cd Entrega1_MCPort

# 2. Crear entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Instalar dependencias
pip install -U pip
pip install -e .
```

> 💡 Alternativa directa:
> ```bash
> pip install -r requirements.txt
> ```

---

## 🧩 Ejecutar en Google Colab

Puedes abrir directamente los notebooks de ejemplo:

👉 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mirdbg/Entrega1_MCPort/blob/main/notebooks/01_Quickstart.ipynb)

O, si clonas el repo desde Colab:

```python
!git clone https://github.com/mirdbg/Entrega1_MCPort.git
%cd Entrega1_MCPort
!pip install -e .
```

---

## 📁 Estructura del proyecto

```
MCPort/
├─ pyproject.toml
├─ requirements.txt
├─ README.md
├─ src/
│  └─ mcport/
│     ├─ __init__.py
│     ├─ models.py
│     ├─ montecarlo.py
│     ├─ providers.py
│     ├─ reports.py
│     └─ utils.py
├─ notebooks/
│  ├─ 01_Quickstart.ipynb
│  ├─ 02_Providers_Yahoo_AlphaVantage.ipynb
│  ├─ 03_Portfolio_Analytics.ipynb
│  ├─ 04_MonteCarlo_Sensitivity.ipynb
│  └─ 05_Reporting.ipynb
├─ tests/
│  ├─ test_models.py
│  ├─ test_utils.py
│  └─ test_montecarlo_extended.py
├─ reports/
│  ├─ figures/
│  └─ pdf/
└─ scripts/
   └─ run_report.py
```

---

## 📊 Ejemplo rápido

```python
from mcport import PriceSeries, Portfolio, MonteCarloSimulation, MonteCarloPlots
import pandas as pd, numpy as np

idx = pd.bdate_range("2023-01-01","2024-12-31")
price = pd.DataFrame({"price": 100*np.exp(np.linspace(0,0.1,len(idx)))}, index=idx)
ps = PriceSeries(symbol="AAPL", asset_type="equity", currency="USD", provider="sim", data=price)

mc = MonteCarloSimulation(price_series=ps)
summ = mc.simulate_and_summarize(days=252, n_sims=1000)

plots = MonteCarloPlots(mc)
plots.plot_history_with_simulations(summ["prices"])
```

---

## 🧠 Módulos principales

| Módulo | Descripción |
|:-------|:-------------|
| `models.py` | Define clases base `PriceSeries` y `Portfolio`. |
| `montecarlo.py` | Implementa simulaciones Monte Carlo tipo GBM. |
| `reports.py` | Crea informes PDF, visualizaciones y resúmenes. |
| `providers.py` | Integraciones con Yahoo Finance y Alpha Vantage. |
| `utils.py` | Funciones de análisis: drawdowns, VaR, CVaR, etc. |

---

## 🧪 Tests

```bash
pytest -v
```

Cubre:
- `PriceSeries` y `Portfolio`
- `drawdowns` y `var_cvar`
- Casos límite en simulaciones Monte Carlo

---

## 📚 Notebooks incluidos

| Notebook | Contenido |
|:----------|:-----------|
| **01_Quickstart** | Pipeline completo (simulación, plots, informe PDF) |
| **02_Providers** | Uso de Yahoo Finance y Alpha Vantage |
| **03_Portfolio_Analytics** | Métricas de rentabilidad y riesgo |
| **04_MonteCarlo_Sensitivity** | Análisis de sensibilidad |
| **05_Reporting** | Exportación de informes PDF |

---

## 🌐 Requisitos

- Python ≥ 3.10  
- pandas, numpy, matplotlib, seaborn, scipy  
- yfinance, alpha_vantage, python-dotenv, pillow

---

## 👩‍💻 Autor

**Miriam del Blanco** 
📍 Madrid, España  
🔗 [LinkedIn](https://www.linkedin.com/in/miriambdelblanco) | [GitHub](https://github.com/mirdbg)

---

## ⚖️ Licencia

Este proyecto está bajo licencia **MIT** — consulta el archivo [`LICENSE`](LICENSE) para más información.
