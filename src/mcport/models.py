from __future__ import annotations
import io
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


from .utils import (
    clean_price_frame,
    to_business_days,
    annualize_stats,
    log_returns,
    sharpe_ratio,
    drawdowns,
    var_cvar,
)

@dataclass
class PriceSeries:
    symbol: str
    asset_type: str = "equity"  # 'equity' | 'index' | 'crypto' | 'fund'
    currency: str = "USD"
    provider: str = "unknown"
    data: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["price"])) #Dataframe con columna 'price' vacía si no se proporciona

    # Estadísiticos básicos media y desviación estándar de los retornos logarítmicos diarios
    mu: float = field(init=False, default=np.nan)
    sigma: float = field(init=False, default=np.nan)

    def __post_init__(self): #Limpieza automática nada más entrar
        self.data = clean_price_frame(self.data)
        # Normaliza a business days con ffill
        self.data = to_business_days(self.data, how="ffill")
        # Calcula ahora los estadísticos básicos
        rets = log_returns(self.data["price"])
        if not rets.empty:
            self.mu = rets.mean()
            self.sigma = rets.std(ddof=1)

    # ---------- Constructores ----------
    @classmethod
    def from_dataframe(cls, symbol: str, df: pd.DataFrame, price_col: str = "price", **meta):
        """Crea PriceSeries desde DataFrame dado. Modifica el nombre de la columna de precio si es necesario."""
        if price_col not in df.columns:
            raise ValueError(f"Se esperaba '{price_col}' en df.")
        df2 = df[[price_col]].rename(columns={price_col: "price"})
        return cls(symbol=symbol, data=df2, **meta)

    # ---------- Core methods ----------
    def clean(self, method: str = "ffill") -> "PriceSeries":
        """Limpieza y reindex a business days por si metemos más datos después."""
        self.data = clean_price_frame(self.data)
        self.data = to_business_days(self.data, how=method)
        return self

    def resample(self, freq: str = "B") -> "PriceSeries":
        """Resample precios a otra frecuencia:'W', 'M'."""
        if self.data.empty:
            return self
        if freq == "B":
            return self
        # Usamos el último día del período como precio representativo
        self.data = self.data.resample(freq).last().dropna()
        # Volvemos a sacar los estadísticos básicos
        rets = self.log_returns()
        if not rets.empty:
            self.mu = rets.mean()
            self.sigma = rets.std(ddof=1)
        return self

    def log_returns(self) -> pd.Series:
        if self.data.empty:
            return pd.Series(dtype=float)
        return log_returns(self.data["price"])

    def extra_stats(self) -> Dict[str, float]:
        """Análisis más completo de estadísticos"""
        r = self.log_returns()
        if r.empty:
            return {k: np.nan for k in ["skew", "kurtosis", "sharpe_daily", "mu_ann", "sigma_ann", "var_95", "cvar_95"]}
        mu_daily = r.mean()
        sigma_daily = r.std(ddof=1)
        mu_ann, sigma_ann = annualize_stats(mu_daily, sigma_daily)
        s = {
            "skew": float(skew(r)),
            "kurtosis": float(kurtosis(r, fisher=True)),  # excess
            "sharpe_daily": float(sharpe_ratio(mu_daily, sigma_daily)),
            "mu_ann": float(mu_ann),
            "sigma_ann": float(sigma_ann),
        }
        v, c = var_cvar(r, alpha=0.05)
        s["var_95"] = float(v)
        s["cvar_95"] = float(c)
        return s

        """

    # ---------- Plot helpers ----------
    def plot_history(self, path: Optional[str] = None):
        if self.data.empty:
            return
        plt.figure()
        self.data["price"].plot(title=f"Historical Price - {self.symbol}")
        if path:
            plt.savefig(path, bbox_inches="tight")
            plt.close()

    def plot_simulations(self, paths: np.ndarray, max_paths: int = 50, path: Optional[str] = None):
        k = min(max_paths, paths.shape[1])
        plt.figure()
        for i in range(k):
            plt.plot(paths[:, i])
        plt.title(f"Monte Carlo Simulations - {self.symbol}")
        if path:
            plt.savefig(path, bbox_inches="tight")
            plt.close()

    def plot_final_hist(self, finals: np.ndarray, bins: int = 50, path: Optional[str] = None):
        plt.figure()
        plt.hist(finals, bins=bins)
        plt.title(f"Terminal Value Distribution - {self.symbol}")
        if path:
            plt.savefig(path, bbox_inches="tight")
            plt.close()
    """
@dataclass
class Portfolio:
    positions: List[PriceSeries]
    weights: List[float]  # si no suma uno, se normalizan automáticamente
    name: str = "Cartera"
    currency: str = "USD"

    # --- Stats mínimas (se rellenan en __post_init__/refresh_stats) ---
    mu_daily: float = field(init=False, default=np.nan)
    sigma_daily: float = field(init=False, default=np.nan)
    mu_ann: float = field(init=False, default=np.nan)
    sigma_ann: float = field(init=False, default=np.nan)
    corr: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)

    def __post_init__(self):
        if len(self.positions) != len(self.weights):
            raise ValueError("Se debe introducir el mismo número de posiciones y pesos.")
        
        w = np.array(self.weights, dtype=float)
        total = w.sum()
        if not np.isclose(total, 1.0):
            # Normalize to avoid errors; warn in report later
            self.weights = (w / total).tolist()
        
        # ➜ calcular estadísticas mínimas al crear
        self._compute_min_stats()

    # ---------- Core methods ----------
    def aligned_prices(self) -> pd.DataFrame:
        """Devuelve df de precios (columna llamada como ticker) en la intersección de las fehcas"""
        frames = []
        for ps in self.positions:
            s = ps.data["price"].rename(ps.symbol)
            frames.append(s)
        df = pd.concat(frames, axis=1, join="inner").dropna().sort_index()
        return df

    def value_series(self, initial_capital: float = 1.0) -> pd.Series:
        """Calcula la serie temporal del valor de la cartera."""
        df = self.aligned_prices()
        if df.empty:
            return pd.Series(dtype=float)
        w = np.array(self.weights)
        # Convertimos precios a log-returns
        rets = np.log(df).diff().dropna()
        # Weighted log-return
        port_log_ret = rets.dot(w)
        # Convert back to price-like equity curve
        eq = np.exp(port_log_ret.cumsum())
        # scale to initial_capital
        eq = initial_capital * eq / eq.iloc[0]
        eq.name = self.name
        return eq

    def log_returns(self) -> pd.Series:
        eq = self.value_series()
        return np.log(eq).diff().dropna()



    # ---------- stats mínimas ----------
    def _compute_min_stats(self) -> None:
        """Calcula y guarda mu/σ diarios y anualizados del PORTFOLIO y corr entre activos."""
        # Stats del portfolio agregado
        r_port = self.log_returns()
        if r_port.empty:
            self.mu_daily = self.sigma_daily = self.mu_ann = self.sigma_ann = np.nan
        else:
            self.mu_daily = float(r_port.mean())
            self.sigma_daily = float(r_port.std(ddof=1))
            self.mu_ann, self.sigma_ann = _annualize(self.mu_daily, self.sigma_daily)

        # Correlación entre activos (útil para Monte Carlo multivariante)
        prices = self.aligned_prices()
        if prices.empty:
            self.corr = pd.DataFrame()
        else:
            self.corr = np.log(prices).diff().dropna().corr()

    def refresh_stats(self) -> None:
        """Recalcula las estadísticas mínimas (llamar tras cambios de posiciones/pesos)."""
        # normaliza por si cambiaron pesos
        w = np.array(self.weights, dtype=float)
        total = w.sum()
        if not np.isclose(total, 1.0):
            self.weights = (w / total).tolist()
        self._compute_min_stats()

    def extra_stats_from_returns(r: pd.Series, rf_daily: float = 0.0) -> dict:
        """
        Estadísticos adicionales a partir de retornos log diarios de PORTFOLIO.
        Devuelve: skew, kurtosis (exceso), sharpe (diario y anual), sortino (anual),
                VaR/CVaR 95% (diarios), max drawdown.
        """
        if r is None or r.empty:
            return {k: np.nan for k in [
                "skew","kurtosis","sharpe_daily","sharpe_annual",
                "sortino_annual","VaR95","CVaR95","max_drawdown"
            ]}

        mu = r.mean()
        sigma = r.std(ddof=1)
        downside = r[r < rf_daily]
        downside_sigma = downside.std(ddof=1) if len(downside) > 0 else np.nan

        # Sharpe
        sharpe_daily = (mu - rf_daily) / sigma if sigma and sigma > 0 else np.nan
        sharpe_annual = sharpe_daily * np.sqrt(252) if not np.isnan(sharpe_daily) else np.nan

        # Sortino (usa solo volatilidad a la baja)
        sortino_daily = (mu - rf_daily) / downside_sigma if downside_sigma and downside_sigma > 0 else np.nan
        sortino_annual = sortino_daily * np.sqrt(252) if not np.isnan(sortino_daily) else np.nan

        # VaR/CVaR al 95% (sobre retornos diarios)
        var95 = np.percentile(r, 5)
        cvar95 = r[r <= var95].mean()

        # Max drawdown (desde la curva de equity)
        eq = np.exp(r.cumsum())
        peak = eq.cummax()
        dd = (eq / peak) - 1.0
        max_dd = float(dd.min())

        return {
            "skew": float(r.skew()),
            "kurtosis": float(r.kurt()),  # exceso (normal = 0)
            "sharpe_daily": float(sharpe_daily) if sharpe_daily == sharpe_daily else np.nan,
            "sharpe_annual": float(sharpe_annual) if sharpe_annual == sharpe_annual else np.nan,
            "sortino_annual": float(sortino_annual) if sortino_annual == sortino_annual else np.nan,
            "VaR95": float(var95),
            "CVaR95": float(cvar95),
            "max_drawdown": max_dd
            }

    def extra_stats_from_portfolio(portfolio: "Portfolio", rf_daily: float = 0.0) -> dict:
        """Convenience: calcula extra stats directamente desde un Portfolio."""
        r = portfolio.log_returns()
        return extra_stats_from_returns(r, rf_daily=rf_daily)       

    """
    # ---------- Reporting ----------
    def report(self, include_warnings: bool = True) -> str:
        
        Generate a Markdown report with stats, warnings, risks, etc.
        
        lines = []
        lines.append(f"# Portfolio Report — {self.name}")
        lines.append(f"- Currency: **{self.currency}**")
        # Composition
        lines.append("## Composition")
        for ps, w in zip(self.positions, self.weights):
            lines.append(f"- {ps.symbol} ({ps.asset_type}, {ps.provider}) — weight: **{w:.2%}**")

        # Price alignment
        df = self.aligned_prices()
        if df.empty:
            lines.append("\n> ⚠️ Not enough overlapping data to compute stats.")
            return "\n".join(lines)

        # Portfolio-level stats
        r = self.log_returns()
        if r.empty:
            lines.append("\n> ⚠️ Not enough returns to compute stats.")
            return "\n".join(lines)

        mu, sigma = r.mean(), r.std(ddof=1)
        mu_ann, sigma_ann = (mu * 252, sigma * np.sqrt(252))
        sr = (mu / sigma) if sigma > 0 else np.nan
        v95, c95 = var_cvar(r, alpha=0.05)

        lines.append("\n## Portfolio Stats (daily)")
        lines.append(f"- Mean (μ): **{mu:.6f}**")
        lines.append(f"- Std (σ): **{sigma:.6f}**")
        lines.append(f"- Sharpe (daily): **{sr:.3f}**")
        lines.append(f"- Annualized μ: **{mu_ann:.3f}**, Annualized σ: **{sigma_ann:.3f}**")
        lines.append(f"- VaR 95% (daily): **{v95:.4f}**, CVaR 95%: **{c95:.4f}**")

        # Component stats
        lines.append("\n## Components (daily)")
        for ps in self.positions:
            stats = ps.extra_stats()
            lines.append(f"### {ps.symbol}")
            lines.append(f"- μ: **{ps.mu:.6f}**, σ: **{ps.sigma:.6f}**, Sharpe(d): **{stats['sharpe_daily']:.3f}**")
            lines.append(f"- Skew: **{stats['skew']:.3f}**, Kurtosis(excess): **{stats['kurtosis']:.3f}**")
            lines.append(f"- VaR95: **{stats['var_95']:.4f}**, CVaR95: **{stats['cvar_95']:.4f}**")
            lines.append("")

        # Warnings
        if include_warnings:
            if not np.isclose(sum(self.weights), 1.0):
                lines.append("> ⚠️ Weights were auto-normalized to sum to 1.")
            if (df <= 0).any().any():
                lines.append("> ⚠️ Non-positive prices detected and filtered.")
            if df.isna().any().any():
                lines.append("> ⚠️ Missing values were forward-filled; results may be sensitive.")
        return "\n".join(lines)

    def plots_report(
        self,
        outdir: str = "./outputs",
        max_paths: int = 50,
        n_sims: int = 1000,
        days: int = 252,
        seed: int = 7
    ) -> Dict[str, str]:
        
        Generate and save a set of useful visualizations. Returns paths to files.
        - Portfolio historical equity curve
        - Drawdown curve
        - Monte Carlo simulated paths (portfolio)
        - Histogram of terminal values (portfolio)
        - Historical price per asset
        
        os.makedirs(outdir, exist_ok=True)
        paths = {}

        # Portfolio history
        eq = self.value_series(initial_capital=1.0)
        if not eq.empty:
            plt.figure()
            eq.plot(title=f"Portfolio Equity - {self.name}")
            p = os.path.join(outdir, f"{self.name}_equity.png")
            plt.savefig(p, bbox_inches="tight"); plt.close()
            paths["portfolio_equity"] = p

            # Drawdowns
            plt.figure()
            drawdowns(eq).plot(title=f"Portfolio Drawdown - {self.name}")
            p = os.path.join(outdir, f"{self.name}_drawdown.png")
            plt.savefig(p, bbox_inches="tight"); plt.close()
            paths["portfolio_drawdown"] = p

        # Portfolio Monte Carlo (aggregate method)
        mc_paths = self.monte_carlo(days=days, n_sims=n_sims, seed=seed, by_components=False)
        plt.figure()
        k = min(max_paths, mc_paths.shape[1])
        for i in range(k):
            plt.plot(mc_paths[:, i])
        plt.title(f"Monte Carlo (Aggregate) - {self.name}")
        p = os.path.join(outdir, f"{self.name}_mc_paths.png")
        plt.savefig(p, bbox_inches="tight"); plt.close()
        paths["portfolio_mc_paths"] = p

        finals = mc_paths[-1, :]
        plt.figure()
        plt.hist(finals, bins=50)
        plt.title(f"Terminal Value Dist (Aggregate) - {self.name}")
        p = os.path.join(outdir, f"{self.name}_mc_term_hist.png")
        plt.savefig(p, bbox_inches="tight"); plt.close()
        paths["portfolio_mc_term_hist"] = p

        # Component price histories
        for ps in self.positions:
            pth = os.path.join(outdir, f"{ps.symbol}_history.png")
            ps.plot_history(path=pth)
            paths[f"{ps.symbol}_history"] = pth

        return paths
        """

