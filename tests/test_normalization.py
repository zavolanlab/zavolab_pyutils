"""
Tests for normalization module.
"""

import pytest
import numpy as np
from zavolab_pyutils.normalization import normalize_by_library_size, compute_library_sizes


class TestNormalization:
    """Test suite for normalization functions."""
    
    @pytest.fixture
    def sample_counts(self):
        """Create sample count matrix for testing."""
        return np.array([
            [100, 200, 150],  # Gene 1
            [50, 100, 80],    # Gene 2
            [200, 400, 300],  # Gene 3
        ])
    
    def test_normalize_by_library_size_default(self, sample_counts):
        """Test normalization with default TMM method."""
        result = normalize_by_library_size(sample_counts)
        assert result is not None
        assert result.shape == sample_counts.shape
    
    def test_normalize_by_library_size_tpm(self, sample_counts):
        """Test normalization with TPM method."""
        result = normalize_by_library_size(sample_counts, method="tpm")
        assert result is not None
    
    def test_normalize_invalid_method(self, sample_counts):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_by_library_size(sample_counts, method="invalid")
    
    def test_compute_library_sizes(self, sample_counts):
        """Test library size computation."""
        lib_sizes = compute_library_sizes(sample_counts)
        expected = np.array([350, 700, 530])
        np.testing.assert_array_equal(lib_sizes, expected)
