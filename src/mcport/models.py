from __future__ import annotations
import io
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

 
@dataclass
class Portfolio:
    positions: List["PriceSeries"]
    weights: List[float]  # si no suma uno, se normalizan automáticamente
    name: str = "Cartera"
    currency: str = "USD"

    # --- Stats mínimas (se rellenan en __post_init__/refresh_stats) ---
    mu_daily: float = field(init=False, default=np.nan)
    sigma_daily: float = field(init=False, default=np.nan)
    mu_ann: float = field(init=False, default=np.nan)
    sigma_ann: float = field(init=False, default=np.nan)
    corr: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)

    # ============================================================
    # INIT
    # ============================================================
    def __post_init__(self):
        if len(self.positions) != len(self.weights):
            raise ValueError("Se debe introducir el mismo número de posiciones y pesos.")
        
        w = np.array(self.weights, dtype=float)
        total = w.sum()
        if not np.isclose(total, 1.0):
            self.weights = (w / total).tolist()
        
        self._compute_min_stats()

    # ============================================================
    # MÉTODOS CORE
    # ============================================================
    def aligned_prices(self) -> pd.DataFrame:
        """Devuelve df de precios (una columna por ticker) en la intersección de fechas"""
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
        rets = np.log(df).diff().dropna()
        port_log_ret = rets.dot(w)
        eq = np.exp(port_log_ret.cumsum())
        eq = initial_capital * eq / eq.iloc[0]
        eq.name = self.name
        return eq

    def log_returns(self) -> pd.Series:
        eq = self.value_series()
        return np.log(eq).diff().dropna()

    # ============================================================
    # STATS MÍNIMAS
    # ============================================================
    def _compute_min_stats(self) -> None:
        """Calcula y guarda mu/σ diarios y anualizados del PORTFOLIO y corr entre activos."""
        r_port = self.log_returns()
        if r_port.empty:
            self.mu_daily = self.sigma_daily = self.mu_ann = self.sigma_ann = np.nan
        else:
            self.mu_daily = float(r_port.mean())
            self.sigma_daily = float(r_port.std(ddof=1))
            self.mu_ann = self.mu_daily * 252
            self.sigma_ann = self.sigma_daily * np.sqrt(252)

        prices = self.aligned_prices()
        if prices.empty:
            self.corr = pd.DataFrame()
        else:
            self.corr = np.log(prices).diff().dropna().corr()

    def refresh_stats(self) -> None:
        """Recalcula las estadísticas mínimas (llamar tras cambios de posiciones/pesos)."""
        w = np.array(self.weights, dtype=float)
        total = w.sum()
        if not np.isclose(total, 1.0):
            self.weights = (w / total).tolist()
        self._compute_min_stats()

    # ============================================================
    # STATS AVANZADAS
    # ============================================================
    @staticmethod
    def extra_stats_from_returns(r: pd.Series, rf_daily: float = 0.0) -> dict:
        """
        Estadísticos adicionales a partir de retornos log diarios de PORTFOLIO.
        Devuelve: skew, kurtosis, sharpe, sortino, VaR/CVaR 95%, max drawdown.
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

        sharpe_daily  = (mu - rf_daily) / sigma if sigma > 0 else np.nan
        sharpe_annual = sharpe_daily * np.sqrt(252) if not np.isnan(sharpe_daily) else np.nan

        sortino_daily = (mu - rf_daily) / downside_sigma if downside_sigma and downside_sigma > 0 else np.nan
        sortino_annual = sortino_daily * np.sqrt(252) if not np.isnan(sortino_daily) else np.nan

        var95  = float(np.percentile(r, 5))
        cvar95 = float(r[r <= var95].mean())

        eq = np.exp(r.cumsum())
        peak = eq.cummax()
        dd = (eq / peak) - 1.0
        max_dd = float(dd.min())

        return {
            "skew": float(r.skew()),
            "kurtosis": float(r.kurt()),
            "sharpe_daily": float(sharpe_daily),
            "sharpe_annual": float(sharpe_annual),
            "sortino_annual": float(sortino_annual),
            "VaR95": var95,
            "CVaR95": cvar95,
            "max_drawdown": max_dd
        }

    def extra_stats_from_portfolio(self, rf_daily: float = 0.0) -> dict:
        """Convenience: calcula extra stats directamente desde el Portfolio."""
        r = self.log_returns()
        return self.extra_stats_from_returns(r, rf_daily=rf_daily)     