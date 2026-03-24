# zavolab_pyutils

Genomic data analysis utilities from the Zavolan Lab. A collection of Python utilities for common bioinformatics tasks including library size normalization, annotation conversion, and other genomic data analysis operations.

## Features

- **Library Size Normalization**: Deseq2-like normalization
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
conda env create -f environment.yml
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
from zavolab_pyutils.read_count_data_analysis import deseq2_normalize

# Create sample count matrix (genes × samples) as a DataFrame
data = {
    "Sample_1": [100, 50, 200],
    "Sample_2": [200, 100, 400],
    "Sample_3": [150, 80, 300],
}
counts_df = pd.DataFrame(data, index=["Gene_1", "Gene_2", "Gene_3"])

# Normalize using DESeq2 method
norm_counts_df, size_factors_df = deseq2_normalize(
    counts_df, 
    sample_list=["Sample_1", "Sample_2", "Sample_3"]
)

print(norm_counts_df)
print(size_factors_df)
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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

## Acknowledgments

Developed by the [Zavolan Lab](https://zavolab.org) at the University of Basel.
