"""
zavolab_pyutils: Genomic data analysis utilities

A collection of utilities for common genomic data analysis tasks including
library size normalization, annotation conversion, visualization, and other bioinformatics operations.
"""

from .read_count_data_analysis import deseq2_normalize
from .annotation import convert_gff_to_gtf, convert_gtf_to_gff, parse_gtf_attributes
from .visualization import plot_size_factors

__version__ = "0.1.0"
__author__ = "Zavolan Lab"
__license__ = "MIT"

__all__ = [
    "deseq2_normalize",
    "convert_gff_to_gtf",
    "convert_gtf_to_gff",
    "parse_gtf_attributes",
    "plot_size_factors",
]
