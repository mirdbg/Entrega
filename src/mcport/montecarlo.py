from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

@dataclass
class MonteCarloSimulation:
    """
    Monte Carlo (GBM) para PriceSeries o Portfolio.
    - Si recibe un PriceSeries: simula un activo individual.
    - Si recibe un Portfolio: simula activos correlacionados ponderados por pesos.
    """

    price_series: object                # PriceSeries o Portfolio
    days: int = 252
    n_sims: int = 2000
    seed: int = 123
    capital_inicial: float = 1000.0
    correlate_assets: bool = True

    _last_paths: Optional[np.ndarray] = field(init=False, default=None)
    _last_values: Optional[np.ndarray] = field(init=False, default=None)
    _last_summary: Optional[Dict[str, Any]] = field(init=False, default=None)

    # --------------------------
    # MÉTODOS PRINCIPALES
    # --------------------------
    def _prepare_inputs(self) -> Dict[str, Any]:
        """Detecta si el input es un PriceSeries o un Portfolio y prepara datos coherentes."""
        obj = self.price_series

        # 🔹 Caso 1: PriceSeries
        if hasattr(obj, "data") and not hasattr(obj, "positions"):
            df_prices = obj.data[["price"]].copy()
            rets = np.log(df_prices["price"]).diff().dropna()
            mu_d = np.array([rets.mean()])
            sigma_d = np.array([rets.std(ddof=1)])
            corr = np.eye(1)
            cov_d = np.outer(sigma_d, sigma_d)
            L = np.linalg.cholesky(cov_d)
            last_prices = np.array([df_prices["price"].iloc[-1]])
            weights = np.array([1.0])
            units = np.array([self.capital_inicial / last_prices[0]])

        # 🔹 Caso 2: Portfolio
        elif hasattr(obj, "positions"):
            df_prices = obj.aligned_prices()
            rets = np.log(df_prices).diff().dropna()
            mu_d = rets.mean().to_numpy(float)
            sigma_d = rets.std(ddof=1).to_numpy(float)
            corr = rets.corr().to_numpy(float) if self.correlate_assets else np.eye(len(mu_d))
            cov_d = np.outer(sigma_d, sigma_d) * corr
            try:
                L = np.linalg.cholesky(cov_d)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(cov_d + 1e-10 * np.eye(len(mu_d)))
            last_prices = df_prices.iloc[-1].to_numpy(float)
            weights = np.array(obj.weights, dtype=float)
            units = (weights * self.capital_inicial) / last_prices

        else:
            raise TypeError("❌ El objeto debe ser PriceSeries o Portfolio válido.")

        return {
            "prices": df_prices,
            "rets": rets,
            "mu_d": mu_d,
            "sigma_d": sigma_d,
            "corr": corr,
            "cov_d": cov_d,
            "L": L,
            "last_prices": last_prices,
            "units": units,
            "weights": weights
        }

    # --------------------------
    # Simulación GBM
    # --------------------------
    def monte_carlo(self) -> Dict[str, Any]:
        rng = np.random.default_rng(self.seed)
        p = self._prepare_inputs()

        n_assets = len(p["mu_d"])
        tray = np.zeros((self.n_sims, self.days + 1, n_assets))
        tray[:, 0, :] = p["last_prices"]

        drift = p["mu_d"] - 0.5 * p["sigma_d"]**2

        for s in range(self.n_sims):
            prices = p["last_prices"].copy()
            for t in range(1, self.days + 1):
                z = rng.standard_normal(n_assets)
                shocks = p["L"] @ z
                prices = prices * np.exp(drift + shocks)
                tray[s, t, :] = prices

        # Si solo hay 1 activo, valores = precios directamente
        if n_assets == 1:
            valores = tray[:, :, 0]
        else:
            valores = np.dot(tray, p["units"])

        self._last_paths = tray
        self._last_values = valores
        self._inputs = p
        return {"trayectorias": tray, "valores": valores, "inputs": p}

    # --------------------------
    # Resumen de resultados
    # --------------------------
    def summarize(self) -> Dict[str, Any]:
        if self._last_values is None:
            raise RuntimeError("Primero ejecuta `monte_carlo()`.")

        vals = self._last_values
        capital0 = float(vals[0, 0]) if vals.ndim > 1 else float(vals[0])
        final_values = vals[:, -1] if vals.ndim > 1 else vals[-1]
        returns = (final_values - capital0) / capital0

        mu_ann = float(self._inputs["mu_d"].mean() * 252)
        sig_ann = float(self._inputs["sigma_d"].mean() * np.sqrt(252))
        var_95 = float(np.percentile(returns, 5))
        cvar_95 = float(returns[returns <= var_95].mean())

        summary = {
            "capital_inicial": self.capital_inicial,
            "mean_final_value": float(final_values.mean()),
            "median_final_value": float(np.median(final_values)),
            "std_final_value": float(final_values.std(ddof=1)),
            "mean_return": float(returns.mean()),
            "VaR_95": var_95,
            "CVaR_95": cvar_95,
            "mu_annualized": mu_ann,
            "sigma_annualized": sig_ann,
        }

        self._last_summary = summary
        return summary

    # --------------------------
    # Ejecución completa
    # --------------------------
    def simulate_and_summarize(self) -> Dict[str, Any]:
        res = self.monte_carlo()
        summary = self.summarize()
        return {
            "trayectorias": res["trayectorias"],
            "valores": res["valores"],
            "inputs": res["inputs"],
            "summary": summary,
        }
