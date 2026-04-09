import pytest
import numpy as np
import pandas as pd
from zavolab_pyutils.read_count_data_analysis import apply_deseq2_normalization

class TestNormalization:
    @pytest.fixture
    def sample_counts(self):
        data = np.array([
            [100, 200, 150],
            [50, 100, 80],
            [200, 400, 300],
        ])
        return pd.DataFrame(data, columns=["S1", "S2", "S3"], index=["G1", "G2", "G3"])

    @pytest.fixture
    def sample_metadata(self):
        return pd.DataFrame({
            "sample": ["S1", "S2", "S3"],
            "condition": ["ctrl", "trt", "trt"]
        })

    def test_apply_deseq2_normalization_shape(self, sample_counts, sample_metadata):
        # Pass the metadata dataframe instead of just a sample list
        norm_counts, sfs_df = apply_deseq2_normalization(
            counts_df=sample_counts, 
            metadata_df=sample_metadata,
            sample_col="sample",
            cond_col="condition"
        )
        assert norm_counts.shape == sample_counts.shape
        assert "sf" in sfs_df.columns