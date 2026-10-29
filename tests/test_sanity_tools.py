import numpy as np
import pandas as pd
import pytest

from zavolab_pyutils.Sanity_tools import (
    SanityNotFoundError,
    apply_sanity_normalization,
    find_sanity_binary,
    run_frac_sanity,
    run_sanity,
    test_differential_expression as sanity_de_test,
    variance_grid,
)
from zavolab_pyutils.Sanity_tools.normalization import _run_fixed_v_batches
from zavolab_pyutils.Sanity_tools.runner import _parse_command, _parse_version

try:
    find_sanity_binary()
    HAS_SANITY = True
except SanityNotFoundError:
    HAS_SANITY = False

needs_sanity = pytest.mark.skipif(
    not HAS_SANITY, reason="Sanity >= 2.0 binary not found (run `make install-sanity`)")


def _poisson_counts(n_genes=60, n_samples=6, lib_size=20_000, seed=0):
    rng = np.random.default_rng(seed)
    props = rng.gamma(shape=0.7, scale=1.0, size=n_genes)
    props /= props.sum()
    counts = rng.poisson(np.outer(props, np.full(n_samples, lib_size)))
    cols = [f"S{i}" for i in range(n_samples)]
    counts_df = pd.DataFrame(counts, index=[f"G{i}" for i in range(n_genes)], columns=cols)
    meta = pd.DataFrame({"sample": cols, "condition": ["A"] * (n_samples // 2) + ["B"] * (n_samples - n_samples // 2)})
    return counts_df, meta


def test_parse_version_and_command():
    assert _parse_version("Sanity version 2.0.0") == (2, 0, 0)
    assert _parse_version("Usage: Sanity <option(s)>") is None
    assert _parse_version("version `GLIBCXX_3.4.32' not found") is None
    params = _parse_command("/bin/Sanity -f x.tsv -d out -vmin 1e-07 -vmax 100.0 -nbin 40 -e 1")
    assert params == {"vmin": 1e-7, "vmax": 100.0, "nbin": 40}
    grid = variance_grid(1e-7, 100.0, 40)
    assert np.isclose(grid[0], 1e-7) and np.isclose(grid[-1], 100.0) and len(grid) == 40


@needs_sanity
def test_run_sanity_keeps_labels_and_drops_empty_genes():
    counts_df, _ = _poisson_counts()
    counts_df.loc["G_empty"] = 0
    out = run_sanity(counts_df, vmin=1e-7, vmax=100, n_threads=1)
    assert "G_empty" not in out.mu.index
    assert list(out.delta.columns) == list(counts_df.columns)
    assert list(out.mu.index) == list(counts_df.index[:-1])
    # LTQ = mu + delta, up to Sanity's six printed decimals
    assert np.allclose(out.ltq, out.delta.add(out.mu, axis=0), atol=2e-6)
    # grid below Sanity's printed precision is reconstructed from the command line
    assert np.isclose(out.v_grid[0], 1e-7) and out.likelihood.shape == (len(out.mu), 160)


@needs_sanity
def test_fixed_v_batches_equal_full_fit():
    counts_df, _ = _poisson_counts(seed=1)
    counts_df = counts_df.loc[counts_df.sum(axis=1) > 0]
    v = 0.05
    full = run_sanity(counts_df, v_method="MAP", vmin=v, vmax=v, nbin=2, n_threads=1)
    batch_index = np.arange(len(counts_df)) % 3
    merged, _ = _run_fixed_v_batches(counts_df, batch_index, np.full(3, v), dict(n_threads=1), None, False)
    for name in ("mu", "delta", "d_delta"):
        assert np.allclose(np.asarray(merged[name]), np.asarray(getattr(full, name)), atol=1e-12)


@needs_sanity
def test_means_on_shared_cpm_scale_across_depths():
    counts_low, meta = _poisson_counts(lib_size=2_000, seed=2)
    counts_high, _ = _poisson_counts(lib_size=200_000, seed=3)
    low = apply_sanity_normalization(counts_low, meta, n_threads=1)
    high = apply_sanity_normalization(counts_high, meta, n_threads=1)
    assert high.median_lib_size / low.median_lib_size > 50
    genes = low.means.index.intersection(high.means.index)
    diff = (high.means.loc[genes, "A"] - low.means.loc[genes, "A"]).mean()
    assert abs(diff) < np.log2(high.median_lib_size / low.median_lib_size) / 4
    # legacy tuple of pySanity's full Bayesian function
    assert len(low.to_tuple()) == 7


@needs_sanity
@pytest.mark.parametrize("empirical_bayes", [False, True])
def test_normalization_outputs(empirical_bayes):
    counts_df, meta = _poisson_counts(n_genes=120, seed=4)
    res = apply_sanity_normalization(counts_df, meta, empirical_bayes=empirical_bayes, n_threads=1)
    genes = res.means.index
    assert list(res.means.columns) == ["A", "B"]
    assert res.norm_counts.shape == (len(genes), 6) and res.variances.shape == (len(genes), 6)
    assert np.all(res.errors_absolute >= res.errors_relative - 1e-12)
    assert np.all(res.vg["v_g"] > 0) and np.all(res.vg["v_g_CI_lower"] <= res.vg["v_g_CI_upper"])
    assert ("trended_prior_v_g" in res.vg) == empirical_bayes
    de = sanity_de_test(res.means, res.errors_relative, "B", "A")
    assert de["p_value"].between(0, 1).all()


@needs_sanity
def test_run_frac_sanity_v1():
    rng = np.random.default_rng(5)
    counts_df, _ = _poisson_counts(n_genes=150, n_samples=12, seed=5)
    # PD samples: a random sub-fraction of each gene
    pd_cols = counts_df.columns[6:]
    counts_df[pd_cols] = rng.binomial(counts_df[pd_cols].to_numpy(), rng.beta(2, 8, size=(150, 1)))
    meta = pd.DataFrame({
        "sample": counts_df.columns,
        "condition": (["UT"] * 3 + ["Stress"] * 3) * 2,
        "fraction": ["Total"] * 6 + ["PD"] * 6,
    })
    res, meta_out, sanity_results = run_frac_sanity(counts_df, meta, version="v1",
                                                    sanity_kwargs=dict(n_threads=1))
    assert set(sanity_results) == {"Total", "PD"}
    assert 0 < meta_out["F_UT"] < 1 and 0 < meta_out["F_Stress"] < 1
    assert len(res) == len(sanity_results["Total"].means.index.intersection(sanity_results["PD"].means.index))
