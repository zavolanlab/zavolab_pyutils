# zavolab_PYutils

Genomic data analysis utilities from the Zavolan Lab. A collection of Python utilities for common bioinformatics tasks including library size normalization, annotation conversion, and other genomic data analysis operations.

## Features

- **Library Size Normalization**: Multiple normalization methods (TMM, TPM, quantile, median)
- **Annotation Conversion**: Convert between GTF and GFF3 formats
- **Genomic Data Processing**: Utilities for working with genomic annotation files

## Installation

### From source
```bash
git clone https://github.com/zavolab/zavolab_PYutils.git
cd zavolab_PYutils
pip install -e .
```

### From conda (after bioconda release)
```bash
conda install -c bioconda zavolab_pyutils
```

### Development installation
```bash
git clone https://github.com/zavolab/zavolab_PYutils.git
cd zavolab_PYutils
pip install -e ".[dev]"
```

## Quick Start

### Library Size Normalization

```python
import numpy as np
from genomic_utils import normalize_by_library_size

# Sample count matrix (genes × samples)
counts = np.array([
    [100, 200, 150],
    [50, 100, 80],
    [200, 400, 300],
])

# Normalize by library size (TMM method)
normalized = normalize_by_library_size(counts, method="tmm")
```

### Annotation Conversion

```python
from genomic_utils import convert_gff_to_gtf

# Convert GFF3 to GTF format
convert_gff_to_gtf("input.gff3", "output.gtf")
```

## Documentation

For detailed documentation, see the [docs](docs/) directory.

## Testing

Run the test suite with pytest:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=src/genomic_utils
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure all tests pass and add new tests for new functionality.

## Citation

If you use zavolab_PYutils in your research, please cite:

```
TODO: Add citation information
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

## Acknowledgments

Developed by the [Zavolan Lab](https://zavolab.org) at the University of Basel.
