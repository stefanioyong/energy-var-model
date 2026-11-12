"""One-day Value at Risk (VaR) and Expected Shortfall (ES), computed three ways.

Conventions used everywhere in this project:
- The position is LONG natural gas, so a loss is a negative return.
- VaR and ES are reported as POSITIVE numbers (a loss), as a fraction of the
  position. A 95% VaR of 0.05 means: on 95% of days the loss should be
  smaller than 5% of the position.
- ES is the average loss on the days that are worse than VaR.
"""

import numpy as np
from scipy import stats


def historical_var_es(returns, alpha):
    """Historical simulation: use the past returns as the loss distribution.

    No distribution is assumed. VaR is the (1 - alpha) quantile of past
    returns, and ES is the average of the returns at or below that quantile.
    """
    r = np.asarray(returns)
    cutoff = np.quantile(r, 1 - alpha)
    var = -cutoff
    es = -r[r <= cutoff].mean()
    return var, es


def normal_var_es(returns, alpha):
    """Parametric VaR assuming returns are normally distributed.

    VaR = -(mu + sigma * z), where z is the (1 - alpha) normal quantile.
    ES  = -(mu - sigma * pdf(z) / (1 - alpha)), the closed-form normal ES.
    """
    r = np.asarray(returns)
    mu, sigma = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(1 - alpha)
    var = -(mu + sigma * z)
    es = -(mu - sigma * stats.norm.pdf(z) / (1 - alpha))
    return var, es


def fit_student_t(returns):
    """Fit a Student-t distribution by maximum likelihood.

    Returns (df, loc, scale). A small df (degrees of freedom) means fat tails.
    """
    df, loc, scale = stats.t.fit(np.asarray(returns))
    return max(df, 2.05), loc, scale  # df must be > 2 for a finite variance


def student_t_var_es(df, loc, scale, alpha):
    """Parametric VaR and ES for a Student-t distribution with fitted parameters.

    The t has fatter tails than the normal, so it usually gives a larger 99% VaR.
    ES uses the closed-form formula for the t distribution.
    """
    q = stats.t.ppf(1 - alpha, df)
    var = -(loc + scale * q)
    tail = stats.t.pdf(q, df) * (df + q**2) / ((df - 1) * (1 - alpha))
    es = -(loc - scale * tail)
    return var, es


def ewma_volatility(returns, lam=0.94):
    """RiskMetrics EWMA volatility: sigma_t^2 = lam * sigma_{t-1}^2 + (1 - lam) * r_{t-1}^2.

    Recent days get more weight, so the estimate reacts quickly when the
    market gets calmer or wilder. Returns an array where entry t is the
    forecast for day t + 1 (it uses returns up to and including day t).
    """
    r = np.asarray(returns)
    var = np.empty(len(r))
    var[0] = r[:30].var()  # start from the variance of the first 30 days
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    # one more step so the last entry is tomorrow's forecast
    next_var = lam * var + (1 - lam) * r**2
    return np.sqrt(next_var)


def monte_carlo_var_es(sigma, shock_params, alpha, n_sims=10_000, rng=None):
    """Monte Carlo VaR and ES: simulate many possible tomorrows and read off the tail.

    Each simulated return is sigma * z:
    - sigma is today's EWMA volatility forecast, so the method reacts to
      volatility clustering (calm periods give a small VaR, wild ones a big VaR);
    - z is a fat-tailed shock drawn from a Student-t (df, loc, scale) that was
      fitted to past standardised returns r_t / sigma_t.
    """
    rng = np.random.default_rng() if rng is None else rng
    df, loc, scale = shock_params
    z = loc + scale * rng.standard_t(df, size=n_sims)
    simulated = sigma * z
    return historical_var_es(simulated, alpha)
