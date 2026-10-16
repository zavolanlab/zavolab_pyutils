"""
fracSanity on top of the C++ Sanity binary: Sanity normalises the Total and the pull-down (PD)
fraction separately, and the fracSanity models of pySanity (v1: `frac_sanity`, v2:
`frac_sanity_v2`) estimate the sub-fraction F and the per-gene sub-fractions alpha_g from the
resulting condition means and standard errors.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from ..pySanity.frac_sanity import calculate_differential_recruitment
from ..pySanity.frac_sanity_v2 import calculate_differential_recruitment_v2
from .normalization import SanityResult, apply_sanity_normalization


def prepare_frac_sanity_inputs(
    total_result: SanityResult,
    pd_result: SanityResult,
    total_condition: str,
    pd_condition: Optional[str] = None,
    genes: Optional[Sequence] = None,
) -> dict:
    """
    fracSanity inputs for one condition, aligned on the genes present in both fractions.

    The LTQs are the condition means (log2 CPM-like) and the variances are the squared relative
    standard errors, as used with pySanity.

    Parameters
    ----------
    total_result, pd_result : SanityResult
        `apply_sanity_normalization` results for the Total and the PD samples.
    total_condition : str
        Condition label in `total_result.means`.
    pd_condition : str or None, optional
        Condition label in `pd_result.means`; None uses `total_condition`.
    genes : sequence or None, optional
        Genes to use, in this order; None uses all genes present in both results.

    Returns
    -------
    dict
        'ltq_total', 'var_total', 'ltq_pd', 'var_pd' (np.ndarray) and 'gene_ids' (list).
    """
    pd_condition = total_condition if pd_condition is None else pd_condition
    if genes is None:
        genes = total_result.means.index.intersection(pd_result.means.index)
    genes = pd.Index(genes)
    return {
        "ltq_total": total_result.means.loc[genes, total_condition].to_numpy(),
        "var_total": total_result.errors_relative.loc[genes, total_condition].to_numpy() ** 2,
        "ltq_pd": pd_result.means.loc[genes, pd_condition].to_numpy(),
        "var_pd": pd_result.errors_relative.loc[genes, pd_condition].to_numpy() ** 2,
        "gene_ids": genes.tolist(),
    }


def run_frac_sanity(
    counts_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    conditions: Sequence[str] = ("UT", "Stress"),
    sample_col: str = "sample",
    cond_col: str = "condition",
    fraction_col: str = "fraction",
    total_label: str = "Total",
    pd_label: str = "PD",
    version: str = "v2",
    sanity_kwargs: Optional[dict] = None,
    **frac_kwargs,
):
    """
    Differential sub-fraction (recruitment) analysis from raw counts of Total and PD samples.

    Runs `apply_sanity_normalization` separately on the Total and the PD samples (the PD fraction
    has an additional source of variability, so the two are not normalised together), then
    fracSanity v1 (`calculate_differential_recruitment`) or v2
    (`calculate_differential_recruitment_v2`) on the genes present in both.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Raw counts, genes as rows, samples of both fractions as columns.
    metadata_df : pd.DataFrame
        One row per sample with `sample_col`, `cond_col` and `fraction_col`.
    conditions : sequence of two str, optional
        Reference and test condition. Their results fill the 'UT' and 'Stress' columns of
        fracSanity's output, respectively. Default ('UT', 'Stress').
    sample_col, cond_col, fraction_col : str, optional
        Column names in `metadata_df`. Defaults 'sample', 'condition', 'fraction'.
    total_label, pd_label : str, optional
        Values of `fraction_col` for Total and PD samples. Defaults 'Total', 'PD'.
    version : {'v2', 'v1'}, optional
        fracSanity version. Default 'v2'.
    sanity_kwargs : dict or None, optional
        Passed to `apply_sanity_normalization` for both fractions (e.g. v_method, vmin, vmax,
        n_threads, empirical_bayes). A given `output_dir` gets the subfolders 'Total' and 'PD'.
    **frac_kwargs
        Passed to the fracSanity function (v1: prior_ut, prior_stress; v2: prior_ut,
        prior_stress and `fit_frac_sanity_v2` options).

    Returns
    -------
    results_df : pd.DataFrame
        Per-gene fracSanity results.
    metadata : dict
        Global fracSanity results (F per condition, ...).
    sanity_results : dict
        {'Total': SanityResult, 'PD': SanityResult}.
    """
    if len(conditions) != 2:
        raise ValueError("conditions must name exactly two conditions (reference, test).")
    sanity_kwargs = dict(sanity_kwargs or {})
    out_root = sanity_kwargs.pop("output_dir", None)

    sanity_results = {}
    for label in (total_label, pd_label):
        meta = metadata_df.loc[metadata_df[fraction_col] == label]
        meta = meta.loc[meta[cond_col].isin(conditions)]
        if meta.empty:
            raise ValueError(f"No samples with {fraction_col} == {label!r} in conditions {list(conditions)}.")
        out = None if out_root is None else f"{out_root}/{label}"
        sanity_results[label] = apply_sanity_normalization(
            counts_df, meta, sample_col=sample_col, cond_col=cond_col, output_dir=out, **sanity_kwargs)

    total, pulldown = sanity_results[total_label], sanity_results[pd_label]
    genes = total.means.index.intersection(pulldown.means.index)
    ref = prepare_frac_sanity_inputs(total, pulldown, conditions[0], genes=genes)
    test = prepare_frac_sanity_inputs(total, pulldown, conditions[1], genes=genes)

    args = dict(
        ltq_total_ut=ref["ltq_total"], var_total_ut=ref["var_total"],
        ltq_pd_ut=ref["ltq_pd"], var_pd_ut=ref["var_pd"],
        ltq_total_stress=test["ltq_total"], var_total_stress=test["var_total"],
        ltq_pd_stress=test["ltq_pd"], var_pd_stress=test["var_pd"],
        gene_ids=ref["gene_ids"],
    )
    if version == "v2":
        results_df, metadata = calculate_differential_recruitment_v2(**args, **frac_kwargs)
    elif version == "v1":
        results_df, metadata = calculate_differential_recruitment(**args, **frac_kwargs)
    else:
        raise ValueError("version must be 'v1' or 'v2'.")
    metadata["conditions"] = {"UT": conditions[0], "Stress": conditions[1]}
    return results_df, metadata, sanity_results
