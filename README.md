# zavolab_pyutils

![Tests](https://github.com/zavolanlab/zavolab_pyutils/actions/workflows/python-app.yml/badge.svg)

Genomic data analysis utilities from the Zavolan Lab. A collection of Python utilities for common bioinformatics tasks including library size normalization, annotation conversion, and other genomic data analysis operations.

## Features

- **Library Size Normalization**: Deseq2-like normalization
- **Mean-Variance Modeling**: Condition-specific overdispersion estimation for RNA-seq counts using Quantile Regression
- **Visualization of expression levels across conditions for selected genes**: Empirical Bayes confidence interval plotting using the Negative Binomial distribution
- **Annotation Conversion**: Convert between GTF and GFF3 formats
- **Genomic Data Processing**: Utilities for working with genomic annotation files

## Installation

### From source
```bash
git clone https://github.com/zavolab/zavolab_pyutils.git
cd zavolab_pyutils
pip install -e .
```

### With conda environment
Create a conda environment from the provided `environment.yml` file:

```bash
conda env create --file=environment.yml
conda activate zavolab_pyutils
```

The environment automatically installs the package and all dependencies including ipykernel for Jupyter notebook support.

### From PyPI (TO DO)
```bash
pip install zavolab_pyutils
```

### From bioconda (TO DO)
```bash
conda install -c bioconda zavolab_pyutils
```

## Quick Start

### Library Size Normalization

```python
import pandas as pd
from zavolab_pyutils.read_count_data_analysis import apply_deseq2_normalization

# 1. Load your count matrix
counts = pd.DataFrame(
    [[100, 200, 150], [50, 100, 80]], 
    columns=["S1", "S2", "S3"], 
    index=["GeneA", "GeneB"]
)

# 2. Define your sample metadata
metadata = pd.DataFrame({
    "sample": ["S1", "S2", "S3"],
    "condition": ["Control", "Treatment", "Treatment"]
})

# 3. Apply DESeq2 median-of-ratios normalization
norm_counts_df, size_factors_df = apply_deseq2_normalization(
    counts_df=counts,
    metadata_df=metadata,
    sample_col="sample",
    cond_col="condition"
)
print(norm_counts_df.head())
print(size_factors_df.head())
```

### Mean-Variance Modeling and Confidence Intervals

```python
from zavolab_pyutils.read_count_data_analysis import model_mean_variance
from zavolab_pyutils.visualization import plot_gene_expression_with_ci

# 1. Define your sample metadata
metadata_df = pd.DataFrame({
    "sample": ["Sample_1", "Sample_2", "Sample_3"],
    "condition": ["Control", "Control", "Treatment"]
})

# 2. Model the condition-specific dispersion (alpha) using Quantile Regression
regr_model_df, plot_data = model_mean_variance(
    norm_counts_df, 
    metadata_df, 
    sample_col='sample', 
    cond_col='condition'
)

# 3. Plot specific genes with Negative Binomial Confidence Intervals
plot_gene_expression_with_ci(
    norm_counts_df, 
    metadata_df, 
    selected_genes=["Gene_1", "Gene_2"], 
    regr_model_df=regr_model_df, 
    savefig_path='./gene_expression_plot.png'
)
```

## Documentation and examples of usage

For detailed documentation, see the [docs](docs/) directory. **TO DO**

For a working example, see [test_module.ipynb](test_module.ipynb) which demonstrates the `deseq2_normalize` function with sample data.

## Testing **TO DO**

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

Please ensure all tests pass and add new tests for new functionality.

## Citation

If you use zavolab_pyutils in your research, please cite:

```
TODO: Add citation information
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

## Acknowledgments

Developed by the Zavolan Lab at the University of Basel.
