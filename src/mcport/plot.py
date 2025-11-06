# src/mcport/plot.py
from __future__ import annotations
import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# Tipos de tu paquete:
# - PriceSeries: .symbol, .data['price'], .log_returns()
# - Portfolio: .name, .positions (List[PriceSeries]), .weights (List[float])
#              .aligned_prices(), .value_series(), .log_returns(), .corr (DataFrame)

# ---------------------------
# Estilo y utilidades comunes
# ---------------------------

_DEF_FIGSIZE = (9, 4.8)
_DPI = 120

def _format_dates(ax):
    try:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    except Exception:
        pass

def _pct(x, pos):
    try:
        return f"{x:.0%}"
    except Exception:
        return str(x)

def _maybe_show_close(fig, show: bool):
    if show:
        try:
            plt.show()
        except Exception:
            pass
        plt.close(fig)

# ---------------------------
# PriceSeries plots
# ---------------------------

def plot_priceseries_history(ps, ax=None, show=True, title=None):
    s = ps.data["price"].dropna()
    if s.empty:
        return None
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_DEF_FIGSIZE, dpi=_DPI)
    else:
        fig = ax.figure

    ax.plot(s.index, s.values, lw=1.6)
    ax.set_title(title or f"Historical Price — {ps.symbol}")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    _format_dates(ax)
    fig.tight_layout()
    _maybe_show_close(fig, show)
    return ax


def plot_priceseries_drawdown(ps, ax=None, show=True, title=None, fill=True):
    s = ps.data["price"].dropna()
    if s.empty:
        return None
    dd = (s / s.cummax()) - 1.0

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_DEF_FIGSIZE, dpi=_DPI)
    else:
        fig = ax.figure

    if fill:
        ax.fill_between(dd.index, dd.values, 0, alpha=0.35, step="pre")
    else:
        ax.plot(dd.index, dd.values, lw=1.2)
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax.set_title(title or f"Drawdown — {ps.symbol}")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    _format_dates(ax)
    fig.tight_layout()
    _maybe_show_close(fig, show)
    return ax


def plot_priceseries_rolling_vol(ps, window: int = 20, ax=None, show=True):
    r = ps.log_returns()
    if r.empty:
        return None
    vol = r.rolling(window).std() * np.sqrt(252)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_DEF_FIGSIZE, dpi=_DPI)
    else:
        fig = ax.figure
    ax.plot(vol.index, vol.values, lw=1.4)
    ax.set_title(f"Rolling Annualized Vol ({window}) — {ps.symbol}")
    ax.set_ylabel("Ann. Vol")
    ax.grid(True, alpha=0.3)
    _format_dates(ax)
    fig.tight_layout()
    _maybe_show_close(fig, show)
    return ax


def plot_priceseries_rolling_return(ps, window: int = 63, ax=None, show=True):
    """Rentabilidad acumulada en ventana (por defecto ~trimestre)."""
    s = ps.data["price"].dropna()
    if s.empty:
        return None
    roll = s.pct_change(window)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_DEF_FIGSIZE, dpi=_DPI)
    else:
        fig = ax.figure
    ax.plot(roll.index, roll.values, lw=1.2)
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax.set_title(f"Rolling Return ({window}) — {ps.symbol}")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)
    _format_dates(ax)
    fig.tight_layout()
    _maybe_show_close(fig, show)
    return ax


def plot_priceseries_returns_hist(ps, bins: int = 50, ax=None, show=True, density=True):
    r = ps.log_returns()
    if r.empty:
        return None

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_DEF_FIGSIZE, dpi=_DPI)
    else:
        fig = ax.figure
    ax.hist(r.values, bins=bins, density=density, alpha=0.8)
    ax.set_title(f"Daily Log-Returns — {ps.symbol}")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density" if density else "Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _maybe_show_close(fig, show)
    return ax

# ---------------------------
# Portfolio plots
# ---------------------------

def plot_portfolio_equity(pf, initial_capital: float = 1.0, ax=None, show=True, title=None):
    eq = pf.value_series(initial_capital=initial_capital)
    if eq.empty:
        return None
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_DEF_FIGSIZE, dpi=_DPI)
    else:
        fig = ax.figure
    ax.plot(eq.index, eq.values, lw=1.6)
    ax.set_title(title or f"Portfolio Equity — {pf.name}")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    _format_dates(ax)
    fig.tight_layout()
    _maybe_show_close(fig, show)
    return ax


def plot_portfolio_drawdown(pf, initial_capital: float = 1.0, ax=None, show=True, fill=True):
    eq = pf.value_series(initial_capital=initial_capital)
    if eq.empty:
        return None
    dd = (eq / eq.cummax()) - 1.0

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_DEF_FIGSIZE, dpi=_DPI)
    else:
        fig = ax.figure
    if fill:
        ax.fill_between(dd.index, dd.values, 0, alpha=0.35, step="pre")
    else:
        ax.plot(dd.index, dd.values, lw=1.2)
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax.set_title(f"Portfolio Drawdown — {pf.name}")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    _format_dates(ax)
    fig.tight_layout()
    _maybe_show_close(fig, show)
    return ax


def plot_portfolio_weights(pf, ax=None, show=True, title=None):
    labels = [ps.symbol for ps in pf.positions]
    w = np.array(pf.weights, dtype=float)
    if w.size == 0:
        return None
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=_DPI)
    else:
        fig = ax.figure
    ax.pie(w, labels=labels, autopct="%1.1f%%", startangle=90, counterclock=False)
    ax.set_title(title or f"Portfolio Weights — {pf.name}")
    fig.tight_layout()
    _maybe_show_close(fig, show)
    return ax


def plot_portfolio_corr_heatmap(pf, ax=None, show=True, title=None, annotate=True):
    prices = pf.aligned_prices()
    if prices.empty:
        return None
    corr = np.log(prices).diff().dropna().corr()

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(6.8, 6.0), dpi=_DPI)
    else:
        fig = ax.figure

    cax = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title(title or f"Return Correlation — {pf.name}")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    if annotate:
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=9)
    fig.tight_layout()
    _maybe_show_close(fig, show)
    return ax


# --- Para MonteCarlo ---

# Espera resultados de tu clase MonteCarloSimulation:
# - simulate_and_summarize() -> dict con keys: 'trayectorias' (sims, T+1, n_assets),
#   'valores' (sims, T+1) para portafolio o (sims, T+1) en un solo activo,
#   y 'inputs' con 'prices' (DataFrame), 'last_prices', etc.

def plot_mc_paths(values: np.ndarray, n_show: int = 100, ax=None, show=True, title="Monte Carlo — Sample Paths"):
    """values: shape (n_sims, T+1)."""
    if values is None or values.size == 0:
        return None
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_DEF_FIGSIZE, dpi=_DPI)
    else:
        fig = ax.figure

    k = min(n_show, values.shape[0])
    for i in range(k):
        ax.plot(values[i, :], lw=0.9, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _maybe_show_close(fig, show)
    return ax


def plot_mc_fan(values: np.ndarray, qs: Sequence[float] = (1, 5, 10, 25, 50, 75, 90, 95, 99),
                ax=None, show=True, title="Monte Carlo — Percentile Fan"):
    """Abanico de percentiles sobre valores simulados (n_sims, T+1)."""
    if values is None or values.size == 0:
        return None
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_DEF_FIGSIZE, dpi=_DPI)
    else:
        fig = ax.figure

    perc = np.percentile(values, q=qs, axis=0)
    t = np.arange(values.shape[1])
    # Rellenamos bandas (ej. 10–90, 25–75) y mediana
    bands = [(10, 90), (25, 75)]
    for lo, hi in bands:
        y1 = np.percentile(values, lo, axis=0)
        y2 = np.percentile(values, hi, axis=0)
        ax.fill_between(t, y1, y2, alpha=0.25, label=f"{lo}-{hi}th pct")

    ax.plot(t, np.percentile(values, 50, axis=0), lw=1.6, label="Median")

    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _maybe_show_close(fig, show)
    return ax


def plot_mc_overlay_with_history(history: pd.Series,
                                 values: np.ndarray,
                                 ax=None, show=True,
                                 title="Historical + MC Paths (overlay)",
                                 last_n_history: Optional[int] = 252,
                                 n_show: int = 60,
                                 freq: str = "B"):  # 👈 añade un parámetro de frecuencia
    h = history.dropna()
    if h.empty or values is None or values.size == 0:
        return None

    if last_n_history is not None:
        h = h.iloc[-last_n_history:]

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_DEF_FIGSIZE, dpi=_DPI)
    else:
        fig = ax.figure

    # Histórico
    ax.plot(h.index, h.values, lw=1.8, label="History")

    # 👇 Construimos fechas simuladas como rango de fechas (no linspace)
    # El primer punto coincide con la última fecha del histórico
    sim_index = pd.date_range(start=h.index[-1], periods=values.shape[1], freq=freq)

    k = min(n_show, values.shape[0])
    for i in range(k):
        ax.plot(sim_index, values[i, :], lw=0.9, alpha=0.6)

    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    _format_dates(ax)
    fig.tight_layout()
    _maybe_show_close(fig, show)
    return ax



def plot_mc_terminal_hist(values: np.ndarray, bins: int = 60, ax=None, show=True, title="Terminal Value Distribution"):
    if values is None or values.size == 0:
        return None
    finals = values[:, -1]

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_DEF_FIGSIZE, dpi=_DPI)
    else:
        fig = ax.figure
    ax.hist(finals, bins=bins, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Terminal Value")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _maybe_show_close(fig, show)
    return ax


def plot_mc_terminal_cdf(values: np.ndarray, ax=None, show=True, title="Terminal Value CDF"):
    if values is None or values.size == 0:
        return None
    finals = np.sort(values[:, -1])
    y = np.linspace(0, 1, finals.shape[0], endpoint=True)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_DEF_FIGSIZE, dpi=_DPI)
    else:
        fig = ax.figure

    ax.plot(finals, y, lw=1.6)
    ax.set_title(title)
    ax.set_xlabel("Terminal Value")
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _maybe_show_close(fig, show)
    return ax
