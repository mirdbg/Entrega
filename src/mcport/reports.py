# mcport/reports.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ✅ Todo viene de plot.py (incluidos los plots de Monte Carlo)
from .plot import (
    # PriceSeries / Portfolio
    plot_priceseries_history,
    plot_priceseries_drawdown,
    plot_priceseries_rolling_vol,
    plot_priceseries_rolling_return,
    plot_priceseries_returns_hist,
    plot_portfolio_equity,
    plot_portfolio_drawdown,
    plot_portfolio_weights,
    plot_portfolio_corr_heatmap,
    # Monte Carlo (ahora también en plot.py)
    plot_mc_paths,
    plot_mc_fan,
    plot_mc_overlay_with_history,
    plot_mc_terminal_hist,
    plot_mc_terminal_cdf,
    # utils
    _DEF_FIGSIZE, _DPI,
)

# (Resto del archivo igual que te lo pasé antes: PDFReportConfig, PortfolioReporter, etc.)


# Espera tus clases:
# - PriceSeries
# - Portfolio
# - MonteCarloSimulation

def _figure_text_page(title: str, lines: Sequence[str], footer: Optional[str] = None):
    fig = plt.figure(figsize=(8.27, 11.69), dpi=_DPI)  # A4 vertical aprox
    fig.suptitle(title, fontsize=18, y=0.98)
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.88])
    ax.axis("off")
    ypos = 0.95
    for line in lines:
        ax.text(0.0, ypos, line, fontsize=11, va="top", ha="left", family="monospace")
        ypos -= 0.035
    if footer:
        ax.text(0.0, 0.02, footer, fontsize=9, va="bottom", ha="left", alpha=0.6)
    return fig


def _table_figure(title: str, df: pd.DataFrame, note: Optional[str] = None):
    """
    Dibuja una tabla centrada ocupando toda la página:
    - Resetea el índice para que NO use rowLabels (que desalinean a la izquierda).
    - Escala y centra la tabla dentro del canvas.
    """
    # A4 vertical aprox.
    fig = plt.figure(figsize=(8.27, 11.69), dpi=_DPI)
    fig.suptitle(title, fontsize=16, y=0.98)

    # Lienzo amplio y centrado
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.88])  # márgenes simétricos
    ax.axis("off")

    # Convertimos el índice en la primera columna para evitar rowLabels
    df_plot = df.reset_index()
    col_labels = list(df_plot.columns)
    cell_text = df_plot.values

    # Construye la tabla centrada
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center")

    # Tipografía y escalado
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    # Intento de auto-anchos por contenido (si tu versión de mpl lo soporta)
    try:
        tbl.auto_set_column_width(col=list(range(len(col_labels))))
    except Exception:
        pass

    # Escala vertical para que quepa mejor; ajusta si tienes muchas filas
    tbl.scale(1.0, 1.2)

    # Nota al pie opcional
    if note:
        ax.text(0.0, 0.02, note, fontsize=9, va="bottom", ha="left", alpha=0.6)

    return fig



@dataclass
class PDFReportConfig:
    include_component_plots: bool = True
    include_corr_heatmap: bool = True
    include_weights: bool = True
    include_returns_hist: bool = True
    rolling_vol_window: int = 20
    rolling_ret_window: int = 63

    # Monte Carlo
    do_montecarlo: bool = True
    mc_days: int = 252
    mc_sims: int = 2000
    mc_seed: int = 123
    mc_show_paths: int = 120
    mc_overlay_history_days: int = 252


class PortfolioReporter:
    """
    Genera un PDF con:
      - Portada
      - Composición de la cartera
      - Stats de cartera y por componente
      - Plots (equity, drawdowns, correlaciones, pesos, y plots por activo)
      - Sección Monte Carlo (overlay histórico + trayectorias, abanico percentiles, histograma terminal, CDF)
    """

    def __init__(self, portfolio, output_path: str, config: Optional[PDFReportConfig] = None):
        self.pf = portfolio
        self.output_path = output_path
        self.cfg = config or PDFReportConfig()

    def _component_stats_df(self) -> pd.DataFrame:
        rows = []
        for ps, w in zip(self.pf.positions, self.pf.weights):
            stats = ps.extra_stats()
            rows.append([
                f"{w:.2%}",
                f"{ps.mu:.6f}",
                f"{ps.sigma:.6f}",
                f"{stats.get('sharpe_daily', np.nan):.3f}",
                f"{stats.get('skew', np.nan):.3f}",
                f"{stats.get('kurtosis', np.nan):.3f}",
                f"{stats.get('var_95', np.nan):.4f}",
                f"{stats.get('cvar_95', np.nan):.4f}",
            ])
        df = pd.DataFrame(rows,
                          index=[ps.symbol for ps in self.pf.positions],
                          columns=["Weight", "μ (daily)", "σ (daily)", "Sharpe(d)", "Skew", "Kurt(excess)", "VaR95", "CVaR95"])
        return df

    def _portfolio_stats_df(self) -> pd.DataFrame:
        r = self.pf.log_returns()
        if r.empty:
            return pd.DataFrame({"Value": []})
        mu, sigma = r.mean(), r.std(ddof=1)
        mu_ann, sigma_ann = mu * 252, sigma * np.sqrt(252)
        sharpe_d = (mu / sigma) if sigma > 0 else np.nan
        # Max drawdown sobre equity:
        eq = self.pf.value_series()
        peak = eq.cummax()
        dd = (eq / peak) - 1.0
        max_dd = dd.min()
        # Simple VaR/CVaR al 95%
        var95 = np.percentile(r.values, 5)
        cvar95 = r[r.values <= var95].mean()

        df = pd.DataFrame({
            "Metric": ["μ (daily)", "σ (daily)", "Sharpe (daily)", "μ (annual)", "σ (annual)", "Max Drawdown", "VaR 95% (daily)", "CVaR 95% (daily)"],
            "Value": [mu, sigma, sharpe_d, mu_ann, sigma_ann, max_dd, var95, cvar95]
        })
        df["Value"] = df["Value"].map(lambda x: f"{x:.6f}" if isinstance(x, (int, float, np.floating)) else x)
        return df.set_index("Metric")

    def build(self) -> str:
        with PdfPages(self.output_path) as pdf:
            # Portada
            lines = [
                f"Portfolio: {self.pf.name}",
                f"Currency: {self.pf.currency}",
                f"Positions: {', '.join([ps.symbol for ps in self.pf.positions])}",
                f"Weights: {', '.join([f'{w:.2%}' for w in self.pf.weights])}",
            ]
            fig = _figure_text_page("Portfolio Report", lines, footer="Generated by mcport.report")
            pdf.savefig(fig); plt.close(fig)

            # Composición
            comp = pd.DataFrame({
                "Symbol": [ps.symbol for ps in self.pf.positions],
                "Asset Type": [ps.asset_type for ps in self.pf.positions],
                "Provider": [ps.provider for ps in self.pf.positions],
                "Weight": [f"{w:.2%}" for w in self.pf.weights],
                "Last Price": [ps.data['price'].dropna().iloc[-1] if not ps.data.empty else np.nan for ps in self.pf.positions],
            })
            comp = comp.set_index("Symbol")
            fig = _table_figure("Composition", comp)
            pdf.savefig(fig); plt.close(fig)

            # Stats de cartera
            fig = _table_figure("Portfolio Stats", self._portfolio_stats_df())
            pdf.savefig(fig); plt.close(fig)

            # Stats por componente
            fig = _table_figure("Component Stats (daily)", self._component_stats_df(),
                                note="μ/σ en retornos log diarios; VaR/CVaR a 95%.")
            pdf.savefig(fig); plt.close(fig)

            # Plots de cartera
            fig = plt.figure(figsize=(8.7, 5.2), dpi=_DPI); ax = fig.add_subplot(111)
            plot_portfolio_equity(self.pf, ax=ax, show=False)
            pdf.savefig(fig); plt.close(fig)

            fig = plt.figure(figsize=(8.7, 5.2), dpi=_DPI); ax = fig.add_subplot(111)
            plot_portfolio_drawdown(self.pf, ax=ax, show=False)
            pdf.savefig(fig); plt.close(fig)

            if self.cfg.include_weights:
                fig = plt.figure(figsize=(7.8, 6.4), dpi=_DPI); ax = fig.add_subplot(111)
                plot_portfolio_weights(self.pf, ax=ax, show=False)
                pdf.savefig(fig); plt.close(fig)

            if self.cfg.include_corr_heatmap:
                fig = plt.figure(figsize=(7.2, 6.4), dpi=_DPI); ax = fig.add_subplot(111)
                plot_portfolio_corr_heatmap(self.pf, ax=ax, show=False)
                pdf.savefig(fig); plt.close(fig)

            # Plots por componente
            if self.cfg.include_component_plots:
                for ps in self.pf.positions:
                    fig = plt.figure(figsize=(8.7, 5.2), dpi=_DPI); ax = fig.add_subplot(111)
                    plot_priceseries_history(ps, ax=ax, show=False)
                    pdf.savefig(fig); plt.close(fig)

                    fig = plt.figure(figsize=(8.7, 5.2), dpi=_DPI); ax = fig.add_subplot(111)
                    plot_priceseries_drawdown(ps, ax=ax, show=False)
                    pdf.savefig(fig); plt.close(fig)

                    fig = plt.figure(figsize=(8.7, 5.2), dpi=_DPI); ax = fig.add_subplot(111)
                    plot_priceseries_rolling_vol(ps, window=self.cfg.rolling_vol_window, ax=ax, show=False)
                    pdf.savefig(fig); plt.close(fig)

                    fig = plt.figure(figsize=(8.7, 5.2), dpi=_DPI); ax = fig.add_subplot(111)
                    plot_priceseries_rolling_return(ps, window=self.cfg.rolling_ret_window, ax=ax, show=False)
                    pdf.savefig(fig); plt.close(fig)

                    if self.cfg.include_returns_hist:
                        fig = plt.figure(figsize=(8.7, 5.2), dpi=_DPI); ax = fig.add_subplot(111)
                        plot_priceseries_returns_hist(ps, ax=ax, show=False)
                        pdf.savefig(fig); plt.close(fig)

            # Sección Monte Carlo
            if self.cfg.do_montecarlo:
                try:
                    from .montecarlo import MonteCarloSimulation
                except Exception:
                    MonteCarloSimulation = None

                if MonteCarloSimulation is not None:
                    mc = MonteCarloSimulation(
                        price_series=self.pf,
                        days=self.cfg.mc_days,
                        n_sims=self.cfg.mc_sims,
                        seed=self.cfg.mc_seed,
                        capital_inicial=1000.0,
                        correlate_assets=True,
                    )
                    out = mc.simulate_and_summarize()
                    values = out["valores"]  # (n_sims, T+1)

                    # Resumen MC en tabla
                    summ = out["summary"]
                    df = pd.DataFrame({
                        "Metric": list(summ.keys()),
                        "Value": list(summ.values()),
                    }).set_index("Metric")
                    fig = _table_figure("Monte Carlo — Summary", df)
                    pdf.savefig(fig); plt.close(fig)

                    # Overlay histórico + trayectorias
                    eq_hist = self.pf.value_series(initial_capital=out["summary"]["capital_inicial"])
                    fig = plt.figure(figsize=(8.7, 5.2), dpi=_DPI); ax = fig.add_subplot(111)
                    plot_mc_overlay_with_history(eq_hist, values, ax=ax, show=False,
                                                 last_n_history=self.cfg.mc_overlay_history_days,
                                                 n_show=self.cfg.mc_show_paths)
                    pdf.savefig(fig); plt.close(fig)

                    # Abanico de percentiles
                    fig = plt.figure(figsize=(8.7, 5.2), dpi=_DPI); ax = fig.add_subplot(111)
                    plot_mc_fan(values, ax=ax, show=False)
                    pdf.savefig(fig); plt.close(fig)

                    # Muestras de trayectorias
                    fig = plt.figure(figsize=(8.7, 5.2), dpi=_DPI); ax = fig.add_subplot(111)
                    plot_mc_paths(values, n_show=self.cfg.mc_show_paths, ax=ax, show=False)
                    pdf.savefig(fig); plt.close(fig)

                    # Distribución terminal
                    fig = plt.figure(figsize=(8.7, 5.2), dpi=_DPI); ax = fig.add_subplot(111)
                    plot_mc_terminal_hist(values, ax=ax, show=False)
                    pdf.savefig(fig); plt.close(fig)

                    fig = plt.figure(figsize=(8.7, 5.2), dpi=_DPI); ax = fig.add_subplot(111)
                    plot_mc_terminal_cdf(values, ax=ax, show=False)
                    pdf.savefig(fig); plt.close(fig)

            # Cierre del PDF añade metadatos
            d = pdf.infodict()
            d["Title"] = f"Portfolio Report — {self.pf.name}"
            d["Author"] = "mcport.report"
            d["Subject"] = "Portfolio Analytics with Monte Carlo"
            d["Keywords"] = "portfolio, monte carlo, risk, analytics, finance"

        return self.output_path


# ---------------------------
# Uso mínimo
# ---------------------------
# from mcport.report import PortfolioReporter, PDFReportConfig
# rep = PortfolioReporter(portfolio, "outputs/portfolio_report.pdf",
#                         PDFReportConfig(do_montecarlo=True, mc_sims=3000))
# path = rep.build()
# print("PDF generado en:", path)
