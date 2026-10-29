"""
fracSanity v2: profile-likelihood estimation of the global sub-fraction F.

Model (as in fracSanity v1):
    alpha_g ~ Beta(a, b),            X_g = log2(alpha_g) <= 0
    D_obs,g = X_g - log2(F) + eps_g,  eps_g ~ measurement error with variance ~ V_g

Differences to v1 (frac_sanity.py):
    1. F is a free parameter. Its profile likelihood is computed on a log2(F) grid,
       re-optimising (a, b) at every grid point. v1 tied F to (a, b) through
       mean/median(2^D), which is dominated by the most PD-enriched genes and does
       not remove the (a, b, F) degeneracy for small F.
    2. The latent constraint alpha_g <= 1 is imposed on X_g only: the likelihood
       integrates over a grid in X that ends exactly at X = 0, and the error kernel
       is integrated exactly over every grid cell. No observed quantity (e.g. D_max)
       enters a constraint; a gene with D_obs + log2(F) > 0 simply sits in the tail
       of its error distribution.
    3. Measurement errors allow for genes whose V_g is underestimated (e.g. genes with
       ~0 reads in Total): by default a contaminated Gaussian,
       (1 - pi) N(0, V_g) + pi N(0, V_g + tau^2), with pi fitted and tau chosen by
       profile likelihood; alternatively Student-t errors with nu estimated. This
       replaces the fixed 10% outlier mixture with a 1e-4 noise floor of v1.
    4. b is unrestricted (b > 0): the divergence of the Beta density at alpha = 1 for
       b < 1 is integrable and harmless once cells are integrated.
    5. F is reported with its identifiability: if the profile likelihood at
       log2F_min (the b -> inf, log-Gamma limit) is within `identifiability_threshold`
       of the optimum, the data alone do not determine F, and the reported F is
       driven by the prior.
    6. Gene-level posteriors of alpha_g are averaged over the posterior of F, so the
       credible intervals reflect both the global F and the gene's own noise.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.special import betainc, betaincc, betaln, gammaincinv, ndtr, stdtr
from scipy.stats import beta as beta_dist
from scipy.stats import kstest, norm
from threadpoolctl import threadpool_limits

LN2 = np.log(2.0)
DEFAULT_NU_GRID = (1.0, 2.0, 3.0, 5.0, 10.0, 30.0, np.inf)

# bounds for (log a, log b) in the inner optimisation
_THETA_BOUNDS = [(np.log(0.02), np.log(500.0)), (np.log(0.02), np.log(1e12))]
# bounds and start for logit(pi), the fraction of genes in the inflated-variance error component;
# pi -> 1 is allowed: it means the V_g of (nearly) all genes are underestimated by tau^2
_LOGIT_PI_BOUNDS = (np.log(1e-5 / (1 - 1e-5)), np.log((1 - 1e-5) / 1e-5))
_LOGIT_PI_START = np.log(0.01 / 0.99)
DEFAULT_TAU_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def beta_prior_from_interval(lower, upper, mass=0.95):
    """
    Beta(alpha_F, beta_F) prior for F whose equal-tailed `mass` interval is [lower, upper].

    Example: glucose starvation, where imaging puts roughly 5-10% of mRNA in P-bodies:
        prior_F = beta_prior_from_interval(0.05, 0.10, mass=0.95)

    Parameters
    ----------
    lower, upper : float
        Interval bounds, 0 < lower < upper < 1.
    mass : float, optional
        Prior probability inside [lower, upper]. Default is 0.95.

    Returns
    -------
    tuple of float
        (alpha_F, beta_F).
    """
    if not (0.0 < lower < upper < 1.0):
        raise ValueError("Require 0 < lower < upper < 1.")
    q_lo, q_hi = (1.0 - mass) / 2.0, 1.0 - (1.0 - mass) / 2.0

    def loss(log_params):
        a, b = np.exp(log_params)
        return ((np.log(beta_dist.ppf(q_lo, a, b)) - np.log(lower)) ** 2
                + (np.log(beta_dist.ppf(q_hi, a, b)) - np.log(upper)) ** 2)

    # moment-matching start: mean at the interval centre, sd from a normal approximation
    m = 0.5 * (lower + upper)
    sd = (upper - lower) / (2.0 * norm.ppf(q_hi))
    k = max(m * (1.0 - m) / sd ** 2 - 1.0, 2.0)
    res = minimize(loss, np.log([m * k, (1.0 - m) * k]), method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-16, "maxiter": 10000})
    a, b = np.exp(res.x)
    return float(a), float(b)


def _log_prior_log2F(log2F, prior_F):
    """Log prior density of log2(F) (Beta prior on F times the Jacobian F ln2)."""
    F = 2.0 ** log2F
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.log(F * LN2)
        if prior_F is not None:
            lp = lp + beta_dist.logpdf(F, prior_F[0], prior_F[1])
    return np.where(np.isfinite(lp), lp, -np.inf)


def _build_kernel(D, sigma, nu, X_lo, h, n_cols, chunk=2000):
    """
    Cell-averaged error densities, Kfull[g, c] = (1/h) * P(eps_g in cell c).

    Column c corresponds to X in [e_j, e_j + h] at log2(F) = f_s with c = j - s + S
    (see `fit_frac_sanity_v2`). The error CDF is integrated exactly, so arbitrarily
    small sigma_g are handled without any loss of probability mass.
    """
    n_genes = len(D)
    K = np.empty((n_genes, n_cols))
    c = np.arange(n_cols + 1) * h
    for start in range(0, n_genes, chunk):
        stop = min(start + chunk, n_genes)
        u = ((D[start:stop] - X_lo)[:, None] - c[None, :]) / sigma[start:stop, None]
        # smaller tail T(-|u|) avoids cancellation far from the centre
        tail = ndtr(-np.abs(u)) if np.isinf(nu) else stdtr(nu, -np.abs(u))
        u_hi, u_lo = u[:, :-1], u[:, 1:]
        t_hi, t_lo = tail[:, :-1], tail[:, 1:]
        cell = np.where(u_lo >= 0.0, t_lo - t_hi,
                        np.where(u_hi <= 0.0, t_hi - t_lo, 1.0 - t_hi - t_lo))
        K[start:stop] = np.clip(cell, 0.0, None) / h
    return K


def _make_cell_prob_fn(e_edges, h, n_exact_top=10):
    """
    Returns w(theta): probabilities of X = log2(alpha) falling into each grid cell
    under Beta(a, b), theta = (log a, log b). The lowest cell absorbs the whole lower
    tail; the cells next to X = 0 use the exact incomplete Beta function, the others
    the midpoint rule.
    """
    x_mid = 0.5 * (e_edges[:-1] + e_edges[1:])
    J = len(x_mid)
    x_mid_ln = x_mid * LN2
    log1m = np.log1p(-(2.0 ** x_mid))
    first_edge = 2.0 ** e_edges[1]
    top_edges = 2.0 ** e_edges[J - n_exact_top:]
    log_const = np.log(LN2) + np.log(h)

    def probs(theta):
        a, b = np.exp(theta[:2])
        with np.errstate(over="ignore", under="ignore"):
            w = np.exp(a * x_mid_ln + (b - 1.0) * log1m - betaln(a, b) + log_const)
        w[0] = betainc(a, b, first_edge)
        sf = betaincc(a, b, top_edges)
        w[J - n_exact_top:] = sf[:-1] - sf[1:]
        w = np.clip(w, 0.0, None)
        total = w.sum()
        return w / total if total > 0 else w

    return probs


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _mixture_lik(Ks, w, theta):
    """Per-gene likelihood and its two components (second is None for one kernel)."""
    L1 = Ks[0] @ w
    if len(Ks) == 1:
        return L1, L1, None
    pi = _expit(theta[2])
    L2 = Ks[1] @ w
    return (1.0 - pi) * L1 + pi * L2, L1, L2


def _optimize_ab(Ks, probs, starts):
    """
    Minimise the NLL at a fixed log2(F) over theta = (log a, log b) for one error kernel,
    or (log a, log b, logit pi) for a two-component (contaminated) error model.
    """
    bounds = _THETA_BOUNDS + ([_LOGIT_PI_BOUNDS] if len(Ks) == 2 else [])

    def fun_and_grad(theta):
        w = probs(theta)
        L, L1, L2 = _mixture_lik(Ks, w, theta)
        L = np.maximum(L, 1e-300)
        inv_L = 1.0 / L
        if len(Ks) == 1:
            r = Ks[0].T @ inv_L
        else:
            pi = _expit(theta[2])
            r = (1.0 - pi) * (Ks[0].T @ inv_L) + pi * (Ks[1].T @ inv_L)
        grad = np.empty(len(theta))
        eps = 1e-6
        for k in range(2):
            tp, tm = theta.copy(), theta.copy()
            tp[k] += eps
            tm[k] -= eps
            grad[k] = -r @ ((probs(tp) - probs(tm)) / (2.0 * eps))
        if len(Ks) == 2:
            grad[2] = -np.sum((L2 - L1) * inv_L) * pi * (1.0 - pi)
        return -np.sum(np.log(L)), grad

    best = None
    for theta0 in starts:
        theta0 = np.asarray(theta0, float)[:len(bounds)]
        if len(theta0) < len(bounds):
            theta0 = np.append(theta0, _LOGIT_PI_START)
        theta0 = np.clip(theta0, [lo for lo, _ in bounds], [hi for _, hi in bounds])
        res = minimize(fun_and_grad, theta0, jac=True, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    return best


def _robust_starts(D, f, a_values=(0.7, 2.0, 5.0)):
    """Starting points whose Beta median matches the median of 2^(D + f)."""
    med = float(np.clip(np.median(2.0 ** (D + f)), 1e-12, 0.5))
    starts = []
    for a in a_values:
        b = max(gammaincinv(a, 0.5) / med, 0.05)
        starts.append(np.log([a, b]))
    return starts


def _profile(K_fulls, probs, D, S, J, h, log2F_min, s_values, store, extra_starts=None):
    """Compute profile NLL at grid indices `s_values`, warm-starting from neighbours."""
    for s in s_values:
        if s in store:
            continue
        f = log2F_min + s * h
        Ks = tuple(K[:, S - s:S - s + J] for K in K_fulls)
        starts = _robust_starts(D, f)
        if store:
            nearest = min(store, key=lambda t: abs(t - s))
            starts.append(store[nearest]["theta"])
        if extra_starts is not None:
            starts.extend(extra_starts)
        res = _optimize_ab(Ks, probs, starts)
        store[s] = {"nll": float(res.fun), "theta": res.x.copy(), "success": bool(res.success)}
    return store


def _quantiles_from_cells(P, e_edges, qs):
    """Quantiles of per-row distributions over cells, linear within cells."""
    cum = np.cumsum(P, axis=1, dtype=np.float64)
    cum /= cum[:, -1:]
    h = e_edges[1] - e_edges[0]
    out = []
    rows = np.arange(P.shape[0])
    for q in qs:
        idx = np.minimum((cum < q).sum(axis=1), P.shape[1] - 1)
        prev = np.where(idx > 0, cum[rows, np.maximum(idx - 1, 0)], 0.0)
        p = np.maximum(cum[rows, idx] - prev, 1e-300)
        out.append(e_edges[idx] + h * np.clip((q - prev) / p, 0.0, 1.0))
    return out


def fit_frac_sanity_v2(
    ltq_total: np.ndarray,
    var_total: np.ndarray,
    ltq_pd: np.ndarray,
    var_pd: np.ndarray,
    prior_F=None,
    error_model="contaminated",
    nu="estimate",
    nu_grid=DEFAULT_NU_GRID,
    tau="estimate",
    tau_grid=DEFAULT_TAU_GRID,
    log2F_min=-20.0,
    grid_step=0.05,
    profile_step=0.1,
    coarse_step=0.5,
    refine_window=25.0,
    identifiability_threshold=1.92,
    keep_gene_posterior=True,
    n_threads=4,
    verbose=False,
):
    """
    Fits fracSanity for one condition by profile likelihood over log2(F).

    Parameters
    ----------
    ltq_total, var_total, ltq_pd, var_pd : np.ndarray
        pySanity LTQ estimates (log2) and their posterior variances for the Total and
        pull-down fractions, aligned by gene.
    prior_F : tuple of float or None, optional
        (alpha_F, beta_F) of a Beta prior on F, e.g. from `beta_prior_from_interval`.
        None means a uniform prior on F (Beta(1, 1)). Default is None.
    error_model : {'contaminated', 'student_t'}, optional
        Measurement error of D_g given the latent X_g.
        'contaminated': eps ~ (1 - pi) N(0, V_g) + pi N(0, V_g + tau^2), i.e. a fitted
        fraction pi of genes whose variance is underestimated by an extra tau^2. Genes
        in the main component keep their nominal pySanity V_g, so their information
        about the boundary alpha <= 1 is preserved.
        'student_t': eps ~ sqrt(V_g) t_nu for all genes.
        Default is 'contaminated'.
    nu : 'estimate' or float, optional
        For error_model='student_t': degrees of freedom; np.inf gives Gaussian errors.
        'estimate' picks the value of `nu_grid` with the highest profile likelihood.
        Default is 'estimate'.
    nu_grid : sequence of float, optional
        Candidate nu values for nu='estimate'.
    tau : 'estimate' or float, optional
        For error_model='contaminated': extra standard deviation (log2 units) of the
        inflated component; pi is always fitted. 'estimate' picks the value of
        `tau_grid` with the highest profile likelihood. Default is 'estimate'.
    tau_grid : sequence of float, optional
        Candidate tau values for tau='estimate'.
    log2F_min : float, optional
        Lower end of the log2(F) grid. Here b is large and the Beta is practically
        its log-Gamma limit, which serves as the reference for identifiability.
        Default is -20.
    grid_step : float, optional
        Width (log2 units) of the integration cells in X = log2(alpha). Default 0.05.
    profile_step, coarse_step : float, optional
        log2(F) spacing of the fine profile (near the optimum / posterior mass) and
        of the coarse profile over the whole range. Multiples of `grid_step`.
    refine_window : float, optional
        Coarse points within this many log-units of the best NLL or log-posterior are
        refined at `profile_step`. Default is 25.
    identifiability_threshold : float, optional
        F counts as identified if the NLL at log2F_min exceeds the minimum by more than
        this. Default 1.92 = chi2_1(0.95) / 2.
    keep_gene_posterior : bool, optional
        Keep the per-gene posterior over X cells (float32, genes x cells). Needed by
        `calculate_differential_recruitment_v2`. Default is True.
    n_threads : int or None, optional
        BLAS threads used during the fit. Many threads on a shared node slow the many
        small matrix-vector products down considerably. None keeps the current setting.
        Default is 4.
    verbose : bool, optional
        Print progress. Default is False.

    Returns
    -------
    dict
        Global estimates (F posterior median 'F', 'F_CI_lower', 'F_CI_upper', 'F_MAP',
        'F_MLE', profile-likelihood interval 'F_profile_CI', 'var_F', 'var_log2_F', 'a',
        'b', 'error_model', 'nu' / 'tau' and 'pi_outlier', per-gene 'outlier_prob'
        (contaminated model), 'F_identifiable', 'delta_nll_loggamma_limit', and
        'delta_nll_at_prior_median': profile NLL at the prior median of F minus its
        minimum, a prior-data conflict measure; values >> 2 mean the data disfavour the
        prior), goodness of fit
        ('ks_stat', 'ks_pval' on PIT values, 'pit'), gene-level posteriors ('alpha_g',
        'alpha_g_CI_lower', 'alpha_g_CI_upper', 'alpha_g_mean', 'log2_alpha_mean',
        'log2_alpha_var', 'alpha_g_plugin'), the profile table 'profile' and grid
        information for downstream use.
    """
    with threadpool_limits(limits=n_threads, user_api="blas"):
        return _fit_frac_sanity_v2(
            ltq_total, var_total, ltq_pd, var_pd, prior_F, error_model, nu, nu_grid,
            tau, tau_grid, log2F_min,
            grid_step, profile_step, coarse_step, refine_window,
            identifiability_threshold, keep_gene_posterior, verbose,
        )


def _fit_frac_sanity_v2(
    ltq_total, var_total, ltq_pd, var_pd, prior_F, error_model, nu, nu_grid,
    tau, tau_grid, log2F_min,
    grid_step, profile_step, coarse_step, refine_window,
    identifiability_threshold, keep_gene_posterior, verbose,
):
    ltq_total, var_total = np.asarray(ltq_total, float), np.asarray(var_total, float)
    ltq_pd, var_pd = np.asarray(ltq_pd, float), np.asarray(var_pd, float)
    D = ltq_pd - ltq_total
    V = np.clip(var_pd + var_total, 1e-8, None)
    if not (np.all(np.isfinite(D)) and np.all(np.isfinite(V))):
        raise ValueError("Non-finite LTQ or variance values; filter genes before fitting.")
    sigma = np.sqrt(V)

    h = float(grid_step)
    k_fine, k_coarse = int(round(profile_step / h)), int(round(coarse_step / h))
    if not (np.isclose(k_fine * h, profile_step) and np.isclose(k_coarse * h, coarse_step)):
        raise ValueError("profile_step and coarse_step must be multiples of grid_step.")
    S = int(round(-log2F_min / h))
    log2F_min = -S * h
    # X grid from X_lo to 0 covers D + log2F for every log2F on the grid
    J = int(np.ceil((-(log2F_min + D.min()) + 10.0) / h))
    X_lo = -J * h
    e_edges = X_lo + h * np.arange(J + 1)
    e_edges[-1] = 0.0
    x_mid = 0.5 * (e_edges[:-1] + e_edges[1:])
    probs = _make_cell_prob_fn(e_edges, h)
    n_cols = J + S

    def log2F_of(s):
        return log2F_min + s * h

    # ---- 1. measurement-error model: candidates are compared by their profile likelihood
    if error_model == "student_t":
        candidates = list(nu_grid) if nu == "estimate" else [float(nu)]
        make_kernels = lambda c: (_build_kernel(D, sigma, c, X_lo, h, n_cols),)
    elif error_model == "contaminated":
        candidates = list(tau_grid) if tau == "estimate" else [float(tau)]
        K_main = _build_kernel(D, sigma, np.inf, X_lo, h, n_cols)
        make_kernels = lambda c: (K_main, _build_kernel(D, np.sqrt(V + c ** 2), np.inf, X_lo, h, n_cols))
    else:
        raise ValueError("error_model must be 'contaminated' or 'student_t'.")
    if len(candidates) > 1:
        k_sel = max(k_coarse, int(round(1.0 / h)))
        s_sel = list(range(S, -1, -k_sel))
        error_scores = {}
        best_c, K_fulls, store = None, None, None
        for c in candidates:
            K_c = make_kernels(c)
            store_c = _profile(K_c, probs, D, S, J, h, log2F_min, s_sel, {})
            error_scores[c] = min(v["nll"] for v in store_c.values())
            if verbose:
                print(f"  {error_model} parameter {c}: min NLL on coarse grid = {error_scores[c]:.2f}")
            if best_c is None or error_scores[c] < error_scores[best_c]:
                best_c, K_fulls, store = c, K_c, store_c
            del K_c
    else:
        best_c, error_scores = candidates[0], None
        K_fulls, store = make_kernels(best_c), {}
    nu_hat = best_c if error_model == "student_t" else np.inf
    tau_hat = best_c if error_model == "contaminated" else np.nan
    if verbose:
        print(f"  using {'nu' if error_model == 'student_t' else 'tau'} = {best_c}")

    # ---- 2. coarse profile over the whole log2(F) range
    s_coarse = sorted(set(range(S, -1, -k_coarse)) | {0, S}, reverse=True)
    _profile(K_fulls, probs, D, S, J, h, log2F_min, s_coarse, store)

    # ---- 3. refine around the likelihood optimum and the posterior mass
    def scores(keys):
        nll = np.array([store[s]["nll"] for s in keys])
        lpost = -nll + _log_prior_log2F(np.array([log2F_of(s) for s in keys]), prior_F)
        return nll, lpost

    keys = sorted(store)
    nll_c, lpost_c = scores(keys)
    keep = (nll_c <= nll_c.min() + refine_window) | (lpost_c >= lpost_c.max() - refine_window)
    s_lo = max(0, min(np.array(keys)[keep]) - k_coarse)
    s_hi = min(S, max(np.array(keys)[keep]) + k_coarse)
    _profile(K_fulls, probs, D, S, J, h, log2F_min, list(range(s_hi, s_lo - 1, -k_fine)), store)

    keys = np.array(sorted(store))
    log2F_grid = np.array([log2F_of(s) for s in keys])
    nll, lpost = scores(keys)
    thetas = np.array([store[s]["theta"] for s in keys])
    successes = np.array([store[s]["success"] for s in keys])

    # ---- 4. posterior of log2(F) (profile likelihood x prior, trapezoid weights)
    widths = np.gradient(log2F_grid) if len(log2F_grid) > 1 else np.ones(1)
    with np.errstate(divide="ignore"):
        lw = lpost - lpost.max() + np.log(widths)
    post = np.exp(lw)
    post /= post.sum()
    cdf = np.cumsum(post)
    q = lambda p: float(np.interp(p, cdf, log2F_grid))
    log2F_median, log2F_lo, log2F_hi = q(0.5), q(0.025), q(0.975)
    mean_log2F = float(post @ log2F_grid)
    var_log2F = float(post @ (log2F_grid - mean_log2F) ** 2)
    mean_F = float(post @ 2.0 ** log2F_grid)
    var_F = float(post @ (2.0 ** log2F_grid - mean_F) ** 2)
    i_map = int(np.argmax(lpost))
    i_mle = int(np.argmin(nll))

    # profile-likelihood interval and identifiability
    inside = nll <= nll.min() + identifiability_threshold
    lo_idx, hi_idx = np.where(inside)[0][[0, -1]]
    prof_lo = 0.0 if lo_idx == 0 else 2.0 ** log2F_grid[lo_idx]
    prof_hi = 1.0 if hi_idx == len(keys) - 1 and keys[-1] == S else 2.0 ** log2F_grid[hi_idx]
    delta_nll_limit = float(nll[0] - nll.min())
    # prior-data conflict: how much worse the likelihood is at the prior median than at the optimum
    delta_nll_prior_median = np.nan
    if prior_F is not None:
        log2F_prior_median = np.log2(beta_dist.ppf(0.5, prior_F[0], prior_F[1]))
        coarse = np.array(sorted(s for s in store if (S - s) % k_coarse == 0))
        nll_coarse = np.array([store[s]["nll"] for s in coarse])
        delta_nll_prior_median = float(
            np.interp(log2F_prior_median, log2F_min + coarse * h, nll_coarse) - nll.min())
    identifiable = bool(delta_nll_limit > identifiability_threshold and lo_idx > 0)
    if not identifiable and prior_F is None:
        warnings.warn(
            "fracSanity v2: the data do not identify F (profile likelihood at the "
            f"log-Gamma limit is within {delta_nll_limit:.2f} of the optimum). "
            "Reported F and alpha_g are driven by the uniform prior on F; supply prior_F.",
            stacklevel=3,
        )

    # ---- 5. gene-level posteriors over X, averaged over the posterior of F
    use = np.where(post > 1e-10 * post.max())[0]
    wts = post[use] / post[use].sum()
    P = np.zeros((len(D), J))
    outlier_prob = np.zeros(len(D))
    for i, wt in zip(use, wts):
        s = keys[i]
        Ks = tuple(K[:, S - s:S - s + J] for K in K_fulls)
        w = probs(thetas[i])
        L, _, L2 = _mixture_lik(Ks, w, thetas[i])
        L = np.maximum(L, 1e-300)
        if len(Ks) == 1:
            P += wt * (Ks[0] * w[None, :]) / L[:, None]
        else:
            pi = _expit(thetas[i][2])
            P += wt * (((1.0 - pi) * Ks[0] + pi * Ks[1]) * w[None, :]) / L[:, None]
            outlier_prob += wt * pi * L2 / L
    row_sum = P.sum(axis=1)
    empty = row_sum <= 0.0
    if np.any(empty):
        # likelihood underflows in every cell (possible with Gaussian errors for genes far
        # outside the grid): place the gene in the cell closest to D + log2F_MAP
        x_target = np.clip(D[empty] + log2F_grid[i_map], X_lo, 0.0)
        j = np.clip(np.searchsorted(e_edges, x_target) - 1, 0, J - 1)
        P[np.where(empty)[0], j] = 1.0
        row_sum = P.sum(axis=1)
    P /= row_sum[:, None]
    log2_alpha_mean = P @ x_mid
    log2_alpha_var = np.clip(P @ x_mid ** 2 - log2_alpha_mean ** 2, 0.0, None)
    alpha_mean = P @ (2.0 ** x_mid)
    x_lo, x_med, x_hi = _quantiles_from_cells(P, e_edges, (0.025, 0.5, 0.975))

    # ---- 6. goodness of fit: PIT of D_obs under the fitted predictive at the MAP
    #         (PIT values of genes in the inflated component are not expected to be uniform
    #         under the main component alone, so the full mixture predictive is used)
    w_map = probs(thetas[i_map])
    f_map = log2F_grid[i_map]
    nz = w_map > 1e-14
    resid = D[:, None] + f_map - x_mid[None, nz]
    if error_model == "student_t":
        cdf_err = ndtr(resid / sigma[:, None]) if np.isinf(nu_hat) else stdtr(nu_hat, resid / sigma[:, None])
        pi_map = 0.0
    else:
        pi_map = float(_expit(thetas[i_map][2]))
        cdf_err = ((1.0 - pi_map) * ndtr(resid / sigma[:, None])
                   + pi_map * ndtr(resid / np.sqrt(V + tau_hat ** 2)[:, None]))
    pit = cdf_err @ (w_map[nz] / w_map[nz].sum())
    ks_stat, ks_pval = kstest(pit, "uniform")

    a_map, b_map = np.exp(thetas[i_map][:2])
    profile_df = pd.DataFrame({
        "log2F": log2F_grid, "F": 2.0 ** log2F_grid, "nll": nll, "log_posterior": lpost,
        "posterior_weight": post, "a": np.exp(thetas[:, 0]), "b": np.exp(thetas[:, 1]),
        "success": successes,
    })
    if error_model == "contaminated":
        profile_df["pi_outlier"] = _expit(thetas[:, 2])

    fit = {
        "F": 2.0 ** log2F_median,
        "F_CI_lower": 2.0 ** log2F_lo, "F_CI_upper": 2.0 ** log2F_hi,
        "F_MAP": 2.0 ** f_map, "F_MLE": 2.0 ** log2F_grid[i_mle],
        "F_profile_CI": (prof_lo, prof_hi),
        "log2F_mean": mean_log2F, "var_log2_F": var_log2F, "F_mean": mean_F, "var_F": var_F,
        "a": float(a_map), "b": float(b_map), "error_model": error_model,
        "nu": nu_hat, "tau": tau_hat, "pi_outlier": pi_map, "error_model_scores": error_scores,
        "outlier_prob": outlier_prob if error_model == "contaminated" else None,
        "F_identifiable": identifiable, "delta_nll_loggamma_limit": delta_nll_limit,
        "delta_nll_at_prior_median": delta_nll_prior_median,
        "nll_min": float(nll.min()),
        "ks_stat": float(ks_stat), "ks_pval": float(ks_pval), "pit": pit,
        "alpha_g": 2.0 ** x_med,
        "alpha_g_CI_lower": 2.0 ** x_lo, "alpha_g_CI_upper": 2.0 ** x_hi,
        "alpha_g_mean": alpha_mean,
        "log2_alpha_mean": log2_alpha_mean, "log2_alpha_var": log2_alpha_var,
        "alpha_g_plugin": 2.0 ** (D + f_map),
        "success": bool(successes[i_map]), "n_inner_failures": int((~successes).sum()),
        "is_mle": prior_F is None, "prior_F": prior_F,
        "profile": profile_df,
        "grid_step": h, "X_edges": e_edges,
    }
    if keep_gene_posterior:
        fit["gene_posterior"] = P.astype(np.float32)
    return fit


def _align_posteriors(P1, P2):
    """Pad the lower end so both posteriors share a grid ending at X = 0."""
    J = max(P1.shape[1], P2.shape[1])
    pad = lambda P: np.pad(P, ((0, 0), (J - P.shape[1], 0))) if P.shape[1] < J else P
    return pad(P1).astype(np.float64), pad(P2).astype(np.float64)


def _delta_log2F_summary(fit_ut, fit_stress, n_samples=200000, seed=0):
    """Posterior of log2(F_stress) - log2(F_ut), assuming independent conditions."""
    rng = np.random.default_rng(seed)
    def draw(fit):
        p = fit["profile"]["posterior_weight"].values
        return rng.choice(fit["profile"]["log2F"].values, size=n_samples, p=p / p.sum())

    d = draw(fit_stress) - draw(fit_ut)
    return {
        "delta_log2_F": float(fit_stress["log2F_mean"] - fit_ut["log2F_mean"]),
        "var_delta_log2_F": float(fit_stress["var_log2_F"] + fit_ut["var_log2_F"]),
        "delta_log2_F_CI": (float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))),
        "prob_delta_log2_F_positive": float(np.mean(d > 0)),
    }


def calculate_differential_recruitment_v2(
    ltq_total_ut, var_total_ut, ltq_pd_ut, var_pd_ut,
    ltq_total_stress, var_total_stress, ltq_pd_stress, var_pd_stress,
    gene_ids, prior_ut=None, prior_stress=None,
    fit_ut=None, fit_stress=None,
    return_gene_posteriors=False,
    **fit_kwargs,
):
    """
    Differential sub-fraction analysis between two conditions with fracSanity v2.

    Mirrors `frac_sanity.calculate_differential_recruitment`, but log2(alpha_g) per
    condition comes from the gene posterior averaged over the posterior of F, so
    intervals and z-scores include the uncertainty of the global F.

    Parameters
    ----------
    ltq_* , var_* : np.ndarray
        pySanity LTQs and posterior variances, aligned by gene, for both conditions.
    gene_ids : list
        Gene identifiers, same order as the arrays.
    prior_ut, prior_stress : tuple of float or None, optional
        Beta priors (alpha_F, beta_F) on F per condition; None = uniform.
    fit_ut, fit_stress : dict or None, optional
        Precomputed outputs of `fit_frac_sanity_v2` (with 'gene_posterior') to reuse a
        condition across comparisons. If None, the condition is fitted here.
    return_gene_posteriors : bool, optional
        Keep the gene x cell posteriors inside the returned fits. Default is False.
    **fit_kwargs
        Passed to `fit_frac_sanity_v2` (e.g. nu, log2F_min, verbose).

    Returns
    -------
    results_df : pandas.DataFrame
        Per gene: posterior mean of log2(alpha) per condition, delta_log2_alpha with
        95% credible interval, its posterior variance, z_score and p_value (normal
        approximation, as in v1), lfsr and prob_increase (exact on the grid),
        alpha posterior medians with 95% credible intervals, plug-in alpha.
    metadata : dict
        'fit_UT', 'fit_Stress', 'F_UT', 'F_Stress', 'delta_log2_F', 'var_delta_log2_F',
        'delta_log2_F_CI', 'prob_delta_log2_F_positive'.
    """
    if fit_ut is None:
        fit_ut = fit_frac_sanity_v2(ltq_total_ut, var_total_ut, ltq_pd_ut, var_pd_ut,
                                    prior_F=prior_ut, **fit_kwargs)
    if fit_stress is None:
        fit_stress = fit_frac_sanity_v2(ltq_total_stress, var_total_stress, ltq_pd_stress,
                                        var_pd_stress, prior_F=prior_stress, **fit_kwargs)
    for fit in (fit_ut, fit_stress):
        if "gene_posterior" not in fit:
            raise ValueError("Precomputed fits must be made with keep_gene_posterior=True.")
    if not np.isclose(fit_ut["grid_step"], fit_stress["grid_step"]):
        raise ValueError("Both fits must use the same grid_step.")
    h = fit_ut["grid_step"]

    # distribution of delta = X_stress - X_ut on a grid of step h (exact convolution)
    P_ut, P_st = _align_posteriors(fit_ut["gene_posterior"], fit_stress["gene_posterior"])
    J = P_ut.shape[1]
    P_delta = np.clip(fftconvolve(P_st, P_ut[:, ::-1], mode="full", axes=1), 0.0, None)
    P_delta /= P_delta.sum(axis=1, keepdims=True)
    centre = J - 1  # index of delta = 0
    prob_increase = P_delta[:, centre + 1:].sum(axis=1) + 0.5 * P_delta[:, centre]
    lfsr = np.minimum(prob_increase, 1.0 - prob_increase)
    delta_edges = (np.arange(2 * J) - centre - 0.5) * h
    d_lo, d_hi = _quantiles_from_cells(P_delta, delta_edges, (0.025, 0.975))

    delta_mean = fit_stress["log2_alpha_mean"] - fit_ut["log2_alpha_mean"]
    delta_var = fit_stress["log2_alpha_var"] + fit_ut["log2_alpha_var"]
    z_scores = delta_mean / np.sqrt(np.maximum(delta_var, 1e-300))

    var_pd_ut, var_total_ut = np.asarray(var_pd_ut), np.asarray(var_total_ut)
    var_pd_stress, var_total_stress = np.asarray(var_pd_stress), np.asarray(var_total_stress)
    results_df = pd.DataFrame({
        "gene_id": gene_ids,
        "log2_alpha_UT": fit_ut["log2_alpha_mean"],
        "log2_alpha_Stress": fit_stress["log2_alpha_mean"],
        "delta_log2_alpha": delta_mean,
        "delta_log2_alpha_CI_lower": d_lo,
        "delta_log2_alpha_CI_upper": d_hi,
        "posterior_variance": delta_var,
        "z_score": z_scores,
        "p_value": 2.0 * norm.sf(np.abs(z_scores)),
        "lfsr": lfsr,
        "prob_increase": prob_increase,
        "var_log2_alpha_UT": fit_ut["log2_alpha_var"],
        "var_log2_alpha_Stress": fit_stress["log2_alpha_var"],
        "var_D_UT": var_pd_ut + var_total_ut,
        "var_D_Stress": var_pd_stress + var_total_stress,
        "alpha_UT": fit_ut["alpha_g"],
        "alpha_UT_CI_lower": fit_ut["alpha_g_CI_lower"],
        "alpha_UT_CI_upper": fit_ut["alpha_g_CI_upper"],
        "alpha_Stress": fit_stress["alpha_g"],
        "alpha_Stress_CI_lower": fit_stress["alpha_g_CI_lower"],
        "alpha_Stress_CI_upper": fit_stress["alpha_g_CI_upper"],
        "alpha_UT_plugin": fit_ut["alpha_g_plugin"],
        "alpha_Stress_plugin": fit_stress["alpha_g_plugin"],
    })

    if not return_gene_posteriors:
        fit_ut = {k: v for k, v in fit_ut.items() if k != "gene_posterior"}
        fit_stress = {k: v for k, v in fit_stress.items() if k != "gene_posterior"}
    metadata = {
        "fit_UT": fit_ut, "fit_Stress": fit_stress,
        "F_UT": fit_ut["F"], "F_Stress": fit_stress["F"],
        **_delta_log2F_summary(fit_ut, fit_stress),
    }
    return results_df, metadata
