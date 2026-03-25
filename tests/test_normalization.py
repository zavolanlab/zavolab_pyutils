import pytest
import numpy as np
import pandas as pd
from zavolab_pyutils.read_count_data_analysis import deseq2_normalize

class TestNormalization:
    @pytest.fixture
    def sample_counts(self):
        # Must return a DataFrame because deseq2_normalize expects one
        data = np.array([
            [100, 200, 150],
            [50, 100, 80],
            [200, 400, 300],
        ])
        return pd.DataFrame(data, columns=["S1", "S2", "S3"], index=["G1", "G2", "G3"])

    def test_deseq2_normalize_shape(self, sample_counts):
        norm_counts, sfs_df = deseq2_normalize(sample_counts, sample_list=["S1", "S2", "S3"])
        assert norm_counts.shape == sample_counts.shape
        assert "sf" in sfs_df.columns