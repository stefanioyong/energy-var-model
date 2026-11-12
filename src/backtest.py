"""Rolling VaR/ES forecasts and the statistical tests used to backtest them."""

import numpy as np
import pandas as pd
from scipy import stats

from . import var_models as vm

METHODS = ["Historical", "Normal", "Student-t", "Monte Carlo (EWMA-t)"]


def rolling_forecasts(returns, alphas=(0.95, 0.99), window=500, refit_every=21,
                      n_sims=10_000, seed=42):
    """Forecast tomorrow's VaR and ES every day, using only data known that day.

    For each day t, the models see the previous `window` returns (about two
    years) and forecast the VaR/ES for day t. Nothing from day t or later is
    used, so the backtest is out of sample.

    The Student-t fit is slow, so it is re-estimated every `refit_every` days
    (about once a month) and held fixed in between.

    Returns a DataFrame indexed by date with the realised return and one
    VaR and ES column per method and confidence level.
    """
    r = returns.to_numpy()
    sigma_ewma = vm.ewma_volatility(r)  # sigma_ewma[t-1] is the forecast for day t
    # standardised returns: z[t] = r[t] / (volatility forecast made the day before)
    z = np.full(len(r), np.nan)
    z[1:] = r[1:] / sigma_ewma[:-1]
    rng = np.random.default_rng(seed)
    rows = []
    for i, t in enumerate(range(window, len(r))):
        past = r[t - window:t]
        if i % refit_every == 0:
            t_params = vm.fit_student_t(past)                 # for the parametric t
            shock_params = vm.fit_student_t(z[max(t - window, 1):t])  # for Monte Carlo shocks
        df, loc, scale = t_params
        row = {"return": r[t]}
        for a in alphas:
            lvl = int(round(a * 100))
            results = {
                "Historical": vm.historical_var_es(past, a),
                "Normal": vm.normal_var_es(past, a),
                "Student-t": vm.student_t_var_es(df, loc, scale, a),
                "Monte Carlo (EWMA-t)": vm.monte_carlo_var_es(sigma_ewma[t - 1], shock_params, a, n_sims, rng),
            }
            for method, (var, es) in results.items():
                row[f"VaR{lvl} {method}"] = var
                row[f"ES{lvl} {method}"] = es
        rows.append(row)
    return pd.DataFrame(rows, index=returns.index[window:])


def kupiec_pof(breaches, alpha):
    """Kupiec proportion-of-failures test.

    Null hypothesis: the true breach probability is (1 - alpha). The
    likelihood-ratio statistic is chi-squared with 1 degree of freedom.
    A small p-value (below 0.05) means the model has too many or too few
    breaches to be trusted.
    """
    hits = np.asarray(breaches, dtype=bool)
    n, x = len(hits), hits.sum()
    p = 1 - alpha
    p_hat = x / n
    log_null = (n - x) * np.log(1 - p) + x * np.log(p)
    log_alt = (n - x) * np.log(1 - p_hat) + x * np.log(p_hat) if 0 < x < n else 0.0
    lr = -2 * (log_null - log_alt)
    return lr, stats.chi2.sf(lr, df=1)


def christoffersen_independence(breaches):
    """Christoffersen independence test.

    Null hypothesis: a breach today does not make a breach tomorrow more
    likely. If breaches arrive in clusters, the model is too slow to react
    to changing volatility, and the p-value will be small.
    """
    h = np.asarray(breaches, dtype=int)
    prev, curr = h[:-1], h[1:]
    n00 = np.sum((prev == 0) & (curr == 0))
    n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0))
    n11 = np.sum((prev == 1) & (curr == 1))
    pi0 = n01 / (n00 + n01)          # P(breach | no breach yesterday)
    pi1 = n11 / (n10 + n11) if (n10 + n11) else 0.0  # P(breach | breach yesterday)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def loglik(count0, count1, prob):
        # count0 * ln(1 - prob) + count1 * ln(prob), with 0 * ln(0) treated as 0
        out = 0.0
        if count0:
            out += count0 * np.log(1 - prob)
        if count1:
            out += count1 * np.log(prob)
        return out

    log_null = loglik(n00 + n10, n01 + n11, pi)
    log_alt = loglik(n00, n01, pi0) + loglik(n10, n11, pi1)
    lr = -2 * (log_null - log_alt)
    return lr, stats.chi2.sf(lr, df=1)


def backtest_table(forecasts, alphas=(0.95, 0.99)):
    """Summarise every method's breaches and test results in one table.

    A breach is a day when the loss is bigger than the VaR forecast
    (return < -VaR). "Avg loss on breach / avg ES" compares the real losses
    on breach days with the ES the model predicted for those days: close to
    1.0 means the ES was about right; above 1.0 means the tail was worse.
    """
    rows = []
    for a in alphas:
        lvl = int(round(a * 100))
        for method in METHODS:
            var = forecasts[f"VaR{lvl} {method}"]
            es = forecasts[f"ES{lvl} {method}"]
            hits = forecasts["return"] < -var
            n, x = len(hits), int(hits.sum())
            lr_pof, p_pof = kupiec_pof(hits, a)
            lr_ind, p_ind = christoffersen_independence(hits)
            p_cc = stats.chi2.sf(lr_pof + lr_ind, df=2)  # conditional coverage
            loss_ratio = (-forecasts["return"][hits]).mean() / es[hits].mean()
            rows.append({
                "Level": f"{lvl}%",
                "Method": method,
                "Days": n,
                "Expected breaches": round(n * (1 - a)),
                "Actual breaches": x,
                "Breach rate": x / n,
                "Kupiec p": p_pof,
                "Christoffersen ind. p": p_ind,
                "Cond. coverage p": p_cc,
                "Avg loss on breach / avg ES": loss_ratio,
            })
    return pd.DataFrame(rows)
