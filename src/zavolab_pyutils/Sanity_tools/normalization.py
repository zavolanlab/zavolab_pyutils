"""
Sanity normalisation of RNA-seq counts with the C++ Sanity binary, summarised as in pySanity:
log2 CPM-like expression per sample, per-condition means and standard errors for differential
expression, per-sample posterior variances, and the gene variances v_g.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.interpolate import interp1d

from .runner import (
    DEFAULT_NBIN,
    DEFAULT_VMAX,
    DEFAULT_VMIN,
    SanityOutput,
    run_sanity,
)

LN2 = np.log(2.0)

# Fixed scale factor that turns Sanity's fraction-of-library output into CPM-like units, so that
# log2 values are comparable across separate runs (as in pySanity)
CPM_SCALE_FACTOR = 1e6

# Sanity prints six decimals, so likelihood values below 5e-7 are written as 0
_LIKELIHOOD_FLOOR = 2.5e-7

_REST_LABEL = "__Sanity_tools_REST__"


@dataclass
class SanityResult:
    """
    Output of `apply_sanity_normalization`. Expression values are log2; v_g, mu and the raw output
    stay in Sanity's natural-log units.

    Attributes
    ----------
    norm_counts : pd.DataFrame
        log2(CPM-like) expression per gene and sample: (mu_g + delta_gc) / ln2 + log2(cpm_scale_factor).
    means : pd.DataFrame
        log2(CPM-like) mean expression per gene and condition.
    errors_relative : pd.DataFrame
        Standard error of each condition mean, without the uncertainty of the gene's global mean
        mu_g. Use these for comparisons between conditions (mu_g cancels in log fold changes).
    errors_absolute : pd.DataFrame
        As `errors_relative`, plus the uncertainty of mu_g. Use these only for comparing a single
        condition mean with an external absolute value.
    vg : pd.DataFrame
        Per gene: 'total_counts', 'mu_log2', 'global_mean_error_log2', 'v_g' (as used for the
        estimates), 'inferred_v_g' (copy of 'v_g', read by `plot_variance_vs_expression`) and
        summaries of the posterior of v_g on Sanity's grid: 'v_g_posterior_mean',
        'v_g_posterior_median', 'v_g_CI_lower', 'v_g_CI_upper' (95%), 'v_g_MAP' (under Sanity's
        1/v prior, or under the empirical Bayes prior), 'v_g_MLE'. With empirical Bayes also
        'trended_prior_v_g'.
    median_lib_size : float
        Median total counts per sample (diagnostic only).
    variances : pd.DataFrame
        Posterior variance (log2 units squared) of each sample's log fold change delta_gc.
    raw : SanityOutput
        Sanity's output from which the estimates were taken.
    raw_pass1 : SanityOutput or None
        With empirical Bayes: the first Sanity run, which provides the likelihood of v_g.
    params : dict
        Settings of the run.
    """

    norm_counts: pd.DataFrame
    means: pd.DataFrame
    errors_relative: pd.DataFrame
    errors_absolute: pd.DataFrame
    vg: pd.DataFrame
    median_lib_size: float
    variances: pd.DataFrame
    raw: SanityOutput = field(repr=False)
    raw_pass1: Optional[SanityOutput] = field(default=None, repr=False)
    params: dict = field(default_factory=dict)

    def to_tuple(self):
        """
        The 7-tuple returned by pySanity's `apply_sanity_normalization_full_bayesian`:
        (norm_counts, means, errors_relative, errors_absolute, vg, median_lib_size, variances).
        """
        return (self.norm_counts, self.means, self.errors_relative, self.errors_absolute,
                self.vg, self.median_lib_size, self.variances)


def summarize_v_posterior(likelihood: pd.DataFrame, log_prior=None) -> pd.DataFrame:
    """
    Summaries of the posterior of v_g on Sanity's log-spaced variance grid.

    Parameters
    ----------
    likelihood : pd.DataFrame
        Genes x grid values, as in `SanityOutput.likelihood` (columns are the grid values).
    log_prior : np.ndarray or None, optional
        Log prior weight of each grid point, genes x grid (or broadcastable). None uses the
        likelihood alone as posterior weights over grid points, which is the 1/v prior Sanity's
        MARG method averages over. Default is None.

    Returns
    -------
    pd.DataFrame
        'v_g_posterior_mean', 'v_g_posterior_median', 'v_g_CI_lower', 'v_g_CI_upper' (2.5% and
        97.5% quantiles), 'v_g_MAP' (most probable grid value; without `log_prior` Sanity's MAP,
        i.e. the maximum of likelihood / v), 'v_g_MLE'.
    """
    v = likelihood.columns.to_numpy(dtype=float)
    lik = likelihood.to_numpy(dtype=float)
    if log_prior is None:
        w = lik
        map_score = lik / v[None, :]
    else:
        logw = np.log(np.maximum(lik, _LIKELIHOOD_FLOOR)) + log_prior
        w = np.exp(logw - logw.max(axis=1, keepdims=True))
        map_score = w
    w = w / w.sum(axis=1, keepdims=True)
    cdf = np.cumsum(w, axis=1)
    quantile = lambda p: v[np.minimum((cdf < p).sum(axis=1), len(v) - 1)]
    return pd.DataFrame({
        "v_g_posterior_mean": w @ v,
        "v_g_posterior_median": quantile(0.5),
        "v_g_CI_lower": quantile(0.025),
        "v_g_CI_upper": quantile(0.975),
        "v_g_MAP": v[np.argmax(map_score, axis=1)],
        "v_g_MLE": v[np.argmax(lik, axis=1)],
    }, index=likelihood.index)


def _empirical_bayes_prior(raw_v, alpha_g, n_samples, shrinkage_strength, loess_variance_threshold_q):
    """
    Log-normal prior on v_g centred on the trend of v_g with expression (as in pySanity).

    Returns the prior mean of log(v_g) per gene, the prior variance of log(v_g) and the trend.
    """
    expr = np.log10(alpha_g)
    valid = raw_v > np.quantile(raw_v, loess_variance_threshold_q)
    valid_expr, valid_vg = expr[valid], raw_v[valid]

    if len(valid_vg) > 10:
        trend = sm.nonparametric.lowess(endog=valid_vg, exog=valid_expr, frac=0.2, return_sorted=False)
        # average the trend over tied expression values before interpolating
        ux, inv = np.unique(valid_expr, return_inverse=True)
        uy = np.bincount(inv, weights=trend) / np.bincount(inv)
        if len(ux) > 1:
            global_trend = interp1d(ux, uy, kind="linear", fill_value="extrapolate")(expr)
        else:
            global_trend = np.full(len(expr), uy[0])
        log_resids = np.log(valid_vg) - np.log(np.clip(trend, 1e-10, None))
        prior_var = np.var(log_resids, ddof=1) if len(log_resids) > 1 else 1.0
    else:
        global_trend = np.full(len(expr), np.median(valid_vg) if len(valid_vg) > 0 else 0.01)
        prior_var = 1.0

    min_floor = np.quantile(valid_vg, 0.05) if len(valid_vg) > 0 else 1e-3
    trended = np.clip(global_trend, min_floor, None)
    # small-sample correction of the trend, as in pySanity
    trended = trended * n_samples / stats.chi2.ppf(0.5, max(1, n_samples - 1))
    prior_log_var = max(prior_var / shrinkage_strength, 0.01)
    return np.log(trended), prior_log_var, trended


def _run_fixed_v_batches(counts, v_index, v_grid, sanity_kwargs, output_dir, overwrite):
    """
    Runs Sanity with v_g fixed per gene: genes sharing a grid value are fitted together, with a
    pseudo-gene that restores every sample's total count. Sanity fits each gene from its own
    counts and the per-sample totals only, so this equals fitting the whole matrix at that v.
    """
    lib_sizes = counts.sum(axis=0)
    parts = {k: [] for k in ("ltq", "ltq_error", "delta", "d_delta", "mu", "d_mu")}
    logs = []
    for k in np.unique(v_index):
        genes = counts.index[v_index == k]
        batch = counts.loc[genes]
        rest = (lib_sizes - batch.sum(axis=0)).clip(lower=0)
        batch = pd.concat([batch, pd.DataFrame([rest.to_numpy()], index=[_REST_LABEL],
                                               columns=counts.columns)])
        out_k = None if output_dir is None else Path(output_dir) / f"eb_pass2_v{k:03d}"
        res = run_sanity(batch, output_dir=out_k, v_method="MAP", vmin=v_grid[k], vmax=v_grid[k],
                         nbin=2, overwrite=overwrite, **sanity_kwargs)
        logs.append(res.log)
        for name in parts:
            parts[name].append(getattr(res, name).drop(index=_REST_LABEL, errors="ignore"))
    merged = {name: pd.concat(dfs).reindex(counts.index) for name, dfs in parts.items()}
    return merged, "\n".join(logs)


def apply_sanity_normalization(
    counts_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    sample_col: str = "sample",
    cond_col: str = "condition",
    v_method: str = "MARG",
    empirical_bayes: bool = False,
    shrinkage_strength: float = 1.0,
    loess_variance_threshold_q: float = 0.1,
    vmin: float = DEFAULT_VMIN,
    vmax: float = DEFAULT_VMAX,
    nbin: int = DEFAULT_NBIN,
    n_threads: int = 4,
    no_norm: bool = False,
    cpm_scale_factor: float = CPM_SCALE_FACTOR,
    output_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    sanity_bin: Optional[str] = None,
    verbose: bool = False,
) -> SanityResult:
    """
    Sanity normalisation of RNA-seq counts with the C++ Sanity binary.

    Replaces pySanity's `apply_sanity_normalization_full_bayesian` (v_method='MARG') and
    `apply_sanity_normalization` (v_method='MLE' or 'MAP'). The per-sample and per-condition
    summaries are computed as in pySanity:

        norm_counts_gc = (mu_g + delta_gc) / ln2 + log2(cpm_scale_factor)
        means_gA       = mu_g / ln2 + log2(cpm_scale_factor) + mean_{c in A} delta_gc / ln2
        SE_rel_gA^2    = [var_{c in A}(delta_gc / ln2) + mean_{c in A}(d_delta_gc^2 / ln2^2)] / n_A
        SE_abs_gA^2    = SE_rel_gA^2 + d_mu_g^2 / ln2^2

    The empirical variance of delta is 0 for conditions with a single sample.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Raw counts, genes as rows, samples as columns.
    metadata_df : pd.DataFrame
        Maps samples (`sample_col`) to conditions (`cond_col`). Only samples present in both
        tables are used; genes without counts in them are dropped.
    sample_col, cond_col : str, optional
        Column names in `metadata_df`. Defaults 'sample', 'condition'.
    v_method : {'MARG', 'MAP', 'EAP', 'MLE'}, optional
        How Sanity treats the gene variance v_g (see `run_sanity`). 'MARG' averages over the
        posterior of v_g, as pySanity's full Bayesian mode. Ignored with empirical_bayes=True.
        Default is 'MARG'.
    empirical_bayes : bool, optional
        Use a gene-specific log-normal prior on v_g centred on the trend of v_g with expression,
        built as in pySanity: a LOWESS fit of the maximum-likelihood v_g against log10 of the
        gene's fraction of all counts, small-sample corrected, with the prior variance taken from
        the residuals. Sanity is run twice: first to get the likelihood of v_g on its grid, then
        with v_g fixed per gene at the most probable grid value under that prior. Unlike pySanity,
        which averages over v_g under the prior, the estimates are therefore conditional on this
        single value. Default is False.
    shrinkage_strength : float, optional
        Divides the prior variance of log(v_g); larger values give a tighter prior. Default 1.0.
    loess_variance_threshold_q : float, optional
        Genes whose maximum-likelihood v_g is at or below this quantile are left out of the trend
        fit. Default 0.1.
    vmin, vmax, nbin : float, float, int, optional
        Grid of v_g (Sanity's -vmin, -vmax, -nbin). Defaults 0.001, 50, 160.
    n_threads : int, optional
        Threads used by Sanity. Default is 4.
    no_norm : bool, optional
        Skip Sanity's normalisation by total counts per sample. Default is False.
    cpm_scale_factor : float, optional
        Scale of the reported log2 expression values. Default 1e6 (counts per million).
    output_dir : str, Path or None, optional
        Keep Sanity's input and output files in this folder (empirical Bayes: pass 1 in
        `output_dir/eb_pass1`, pass 2 in `output_dir/eb_pass2_v<grid index>`). None uses
        temporary folders. Default is None.
    overwrite : bool, optional
        Allow overwriting Sanity output in `output_dir`. Default is False.
    sanity_bin : str or None, optional
        Path to the Sanity binary; None searches SANITY_BIN, the conda env and PATH.
    verbose : bool, optional
        Print Sanity's console output. Default is False.

    Returns
    -------
    SanityResult
        Use `.to_tuple()` for pySanity's 7-tuple.
    """
    if metadata_df[sample_col].duplicated().any():
        raise ValueError(f"Duplicate entries found in metadata column '{sample_col}'.")
    sample_map = metadata_df.set_index(sample_col)[cond_col]
    common_samples = counts_df.columns.intersection(sample_map.index)
    if len(common_samples) == 0:
        raise ValueError("No samples of counts_df are listed in metadata_df.")
    sample_map = sample_map[common_samples]
    counts = counts_df[common_samples]
    counts = counts.loc[counts.to_numpy(dtype=float).sum(axis=1) > 0]
    n_samples = counts.shape[1]
    if empirical_bayes and _REST_LABEL in counts.index:
        raise ValueError(f"Gene name {_REST_LABEL!r} is reserved.")

    sanity_kwargs = dict(n_threads=n_threads, no_norm=no_norm, sanity_bin=sanity_bin, verbose=verbose)
    grid_kwargs = dict(vmin=vmin, vmax=vmax, nbin=nbin)
    params = dict(v_method=v_method.upper(), empirical_bayes=empirical_bayes, cpm_scale_factor=cpm_scale_factor,
                  **grid_kwargs, no_norm=no_norm, n_samples=n_samples)

    if not empirical_bayes:
        raw = run_sanity(counts, output_dir=output_dir, v_method=v_method, overwrite=overwrite,
                         **grid_kwargs, **sanity_kwargs)
        raw_pass1 = None
        v_post = summarize_v_posterior(raw.likelihood)
        v_used = raw.variance
        extra_vg = {}
    else:
        out1 = None if output_dir is None else Path(output_dir) / "eb_pass1"
        raw_pass1 = run_sanity(counts, output_dir=out1, v_method="MAP", overwrite=overwrite,
                               **grid_kwargs, **sanity_kwargs)
        v_grid = raw_pass1.v_grid
        lik = raw_pass1.likelihood
        raw_v_mle = v_grid[np.argmax(lik.to_numpy(), axis=1)]
        alpha_g = counts.sum(axis=1).to_numpy(dtype=float) / counts.to_numpy(dtype=float).sum()
        prior_mean, prior_var, trended = _empirical_bayes_prior(
            raw_v_mle, alpha_g, n_samples, shrinkage_strength, loess_variance_threshold_q)
        log_prior = -0.5 * (np.log(v_grid)[None, :] - prior_mean[:, None]) ** 2 / prior_var
        v_post = summarize_v_posterior(lik, log_prior=log_prior)
        v_index = np.searchsorted(v_grid, v_post["v_g_MAP"].to_numpy())

        merged, log = _run_fixed_v_batches(counts, v_index, v_grid, sanity_kwargs, output_dir, overwrite)
        v_used = pd.Series(v_grid[v_index], index=counts.index, name="variance")
        raw = SanityOutput(
            likelihood=lik, v_grid=v_grid, variance=v_used, v_method="EB-MAP",
            version=raw_pass1.version, command=raw_pass1.command, output_dir=None, log=log,
            **merged,
        )
        extra_vg = {"trended_prior_v_g": trended}
        params.update(shrinkage_strength=shrinkage_strength, prior_log_var=prior_var,
                      loess_variance_threshold_q=loess_variance_threshold_q)

    genes = raw.mu.index
    offset = np.log2(cpm_scale_factor)
    mu_log2 = raw.mu / LN2
    deltas_log2 = raw.delta / LN2
    variances_log2 = raw.d_delta ** 2 / LN2 ** 2
    global_error_var_log2 = raw.d_mu ** 2 / LN2 ** 2

    norm_counts = deltas_log2.add(mu_log2 + offset, axis=0)

    means, errors_relative, errors_absolute = {}, {}, {}
    for cond in sample_map.unique():
        samples = sample_map.index[sample_map == cond]
        n_reps = len(samples)
        d = deltas_log2[samples]
        means[cond] = mu_log2 + offset + d.mean(axis=1)
        empirical_var = d.var(axis=1, ddof=1) if n_reps > 1 else 0.0
        posterior_var = variances_log2[samples].mean(axis=1)
        relative_var = (empirical_var + posterior_var) / n_reps
        errors_relative[cond] = np.sqrt(relative_var)
        errors_absolute[cond] = np.sqrt(relative_var + global_error_var_log2)

    vg = pd.DataFrame({
        "total_counts": counts.loc[genes].sum(axis=1),
        "mu_log2": mu_log2 + offset,
        "global_mean_error_log2": np.sqrt(global_error_var_log2),
        "v_g": v_used.to_numpy(),
        "inferred_v_g": v_used.to_numpy(),
    }, index=genes)
    vg = pd.concat([vg, v_post], axis=1)
    for name, values in extra_vg.items():
        vg[name] = values

    return SanityResult(
        norm_counts=norm_counts,
        means=pd.DataFrame(means, index=genes),
        errors_relative=pd.DataFrame(errors_relative, index=genes),
        errors_absolute=pd.DataFrame(errors_absolute, index=genes),
        vg=vg,
        median_lib_size=float(np.median(counts.sum(axis=0))),
        variances=variances_log2,
        raw=raw,
        raw_pass1=raw_pass1,
        params=params,
    )
