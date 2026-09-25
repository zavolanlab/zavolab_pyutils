import pytest
import numpy as np
import pandas as pd
from zavolab_pyutils.pySanity.sanity import apply_sanity_normalization_full_bayesian


class TestSanityCPMScale:
    @pytest.fixture
    def base_proportions(self):
        rng = np.random.default_rng(0)
        raw = rng.gamma(shape=2.0, scale=1.0, size=15)
        return raw / raw.sum()

    def _make_counts(self, proportions, total_lib_size, n_samples, seed):
        rng = np.random.default_rng(seed)
        n_genes = len(proportions)
        lam = np.outer(proportions, np.full(n_samples, total_lib_size))
        counts = rng.poisson(lam=lam)
        cols = [f"S{i}" for i in range(n_samples)]
        idx = [f"G{i}" for i in range(n_genes)]
        counts_df = pd.DataFrame(counts, index=idx, columns=cols)
        metadata_df = pd.DataFrame({"sample": cols, "condition": ["cond"] * n_samples})
        return counts_df, metadata_df

    def test_means_on_shared_cpm_scale_across_separate_calls(self, base_proportions):
        # Two separate calls (mimicking e.g. Total vs. PD fractions) drawn from the
        # SAME underlying relative gene abundances, but at very different sequencing
        # depths -> very different median_lib_size per call.
        counts_low, meta_low = self._make_counts(base_proportions, total_lib_size=2_000, n_samples=4, seed=1)
        counts_high, meta_high = self._make_counts(base_proportions, total_lib_size=200_000, n_samples=4, seed=2)

        _, means_low, _, _, _, median_lib_size_low, _ = apply_sanity_normalization_full_bayesian(
            counts_df=counts_low, metadata_df=meta_low, sample_col="sample", cond_col="condition", n_cores=1,
        )
        _, means_high, _, _, _, median_lib_size_high, _ = apply_sanity_normalization_full_bayesian(
            counts_df=counts_high, metadata_df=meta_high, sample_col="sample", cond_col="condition", n_cores=1,
        )

        # Confirm the two calls really do have very different sequencing depths.
        assert median_lib_size_high / median_lib_size_low > 50

        # Despite that, since both are now rescaled by the SAME fixed
        # cpm_scale_factor (not by their own differing median_lib_size), per-gene
        # mean log2(CPM) values from the two independent calls should land on the
        # same absolute scale (up to inference noise) -- not offset by
        # log2(median_lib_size_high / median_lib_size_low), which is the bug this
        # change fixes.
        common_genes = means_low.index.intersection(means_high.index)
        diff = (means_high.loc[common_genes, "cond"] - means_low.loc[common_genes, "cond"]).mean()

        old_bug_offset = np.log2(median_lib_size_high / median_lib_size_low)
        assert abs(diff) < old_bug_offset / 4

    def test_cpm_scale_factor_is_configurable(self, base_proportions):
        counts_df, metadata_df = self._make_counts(base_proportions, total_lib_size=5_000, n_samples=3, seed=3)

        _, means_default, _, _, _, _, _ = apply_sanity_normalization_full_bayesian(
            counts_df=counts_df, metadata_df=metadata_df, sample_col="sample", cond_col="condition", n_cores=1,
        )
        _, means_custom, _, _, _, _, _ = apply_sanity_normalization_full_bayesian(
            counts_df=counts_df, metadata_df=metadata_df, sample_col="sample", cond_col="condition", n_cores=1,
            cpm_scale_factor=1e4,
        )

        expected_shift = np.log2(1e4) - np.log2(1e6)
        diff = (means_custom["cond"] - means_default["cond"]).mean()
        assert diff == pytest.approx(expected_shift, abs=1e-9)
