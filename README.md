# 🧠 MCPort — Monte Carlo Portfolio Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest%20passing-brightgreen.svg)](#-tests)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

> **MCPort** es un toolkit educativo y financiero para simular, analizar y visualizar carteras de inversión mediante métodos de **Monte Carlo**.  
> Incluye módulos para extracción de precios, cálculo de métricas, generación de informes PDF y visualizaciones.

---

## 🚀 Instalación local

# 1 Clonar el repositorio
git clone https://github.com/mirdbg/Entrega1_MCPort.git
cd Entrega1_MCPort

# 2 (Opcional) Crear y activar un entorno virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3 Instalar dependencias
pip install -U pip
pip install -r requirements.txt
# o modo editable si quieres importar mcport desde src/
pip install -e .

---

## 🧩 Ejecutar en Google Colab

Puedes abrir directamente los notebooks de ejemplo:

👉 [![Open In Colab Quickstart](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mirdbg/Entrega1_MCPort/blob/main/notebooks/01_Quickstart.ipynb)

👉 [![Open In Colab Quickstart Report](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mirdbg/Entrega1_MCPort/blob/main/notebooks/01_Quickstart.ipynb)

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
│     ├─ plot.py
│     ├─ providers.py
│     ├─ reports.py
│     └─ utils.py
└─ notebooks/
   ├─ 01_Quickstart.ipynb
   └─ 02_Reporting.ipynb
   └─ plot
   └─ reports

```

---

## 📊 Ejemplo rápido

```python
from mcport import PriceSeries, Portfolio, MonteCarloSimulation  # y/o MonteCarloPlots
import pandas as pd, numpy as np

# Serie sintética de precios
idx = pd.bdate_range("2023-01-01", "2024-12-31")
price = pd.DataFrame({"price": 100*np.exp(np.linspace(0, 0.1, len(idx)))}, index=idx)

ps = PriceSeries(symbol="AAPL", asset_type="equity", currency="USD", provider="sim", data=price)

mc = MonteCarloSimulation(price_series=ps)
summary = mc.simulate_and_summarize(days=252, n_sims=1000)

```

---

## 🧠 Módulos principales

| Módulo | Descripción |
|:-------|:-------------|
| `models.py` | Define clases base `PriceSeries` y `Portfolio`. |
| `montecarlo.py` | Implementa simulaciones Monte Carlo tipo GBM. |
| `plot.py` | Crea visualizaciones analizando PriceSeries, Portfolios y Montecarlo. |
| `reports.py` | Crea informes PDF. |
| `providers.py` | Integraciones con Yahoo Finance y Alpha Vantage. |
| `utils.py` | Funciones de análisis: drawdowns, VaR, CVaR, etc. |

---

## 📚 Notebooks incluidos

| Notebook | Contenido |
|:----------|:-----------|
| **01_Quickstart** | Pipeline completo (simulación, plots, informe PDF) |
| **02_Reporting** | Exportación de informes PDF |

---

## 🌐 Requisitos

- Python ≥ 3.10  
- pandas, numpy, matplotlib, seaborn, scipy  
- yfinance, alpha_vantage, python-dotenv, pillow

---

## 👩‍💻 Autor

**Miriam del Blanco** 
📍 Madrid, España  
🔗 [LinkedIn](https://www.linkedin.com/in/miriam-del-blanco-gonz%C3%A1lez/) | [GitHub](https://github.com/mirdbg)

---

## ⚖️ Licencia

Este proyecto está bajo licencia **MIT** — consulta el archivo [`LICENSE`](LICENSE) para más información.
