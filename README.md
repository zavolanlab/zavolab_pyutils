[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19483710.svg)](https://doi.org/10.5281/zenodo.19483710)
![Tests](https://github.com/zavolanlab/zavolab_pyutils/actions/workflows/python-app.yml/badge.svg)
# zavolab_pyutils

Genomic data analysis utilities from the Zavolan Lab. A collection of Python utilities for common bioinformatics tasks including library size normalization, annotation conversion, and other genomic data analysis operations.

## Features

- **Library Size Normalization**: [Deseq2-like](https://pubmed.ncbi.nlm.nih.gov/25516281/) normalization, [Sanity-like](https://pubmed.ncbi.nlm.nih.gov/33927416/) normalization, termed **pySanity**
- **fracSanity**: pySanity-based modelling of the RNA-seq-like data consisting of pull-down (sub-cellular fraction) and total fractions, and two conditions (for simlicity referred to as UT and Stress)
- **Mean-Variance Modeling**: Condition-specific overdispersion estimation for RNA-seq counts using Quantile Regression
- **Visualization of expression levels across conditions for selected genes**: confidence interval plotting based on pySanity outputs or Quantile regression
- **Visualization of isoform relative usage levels across conditions for selected isoforms**: confidence interval plotting based on pySanity outputs
- **Differential expression and differential usage analysis**: based on pySanity outputs
- **Annotation Conversion and Processing**: Convert between GTF and GFF3 formats, extract terminal exons from annotation etc.
- **CLI pythonic functions** used in various Snakemake/Nextflow pipelines of the Zavolan lab.
## Installation

### Developer Setup from source, with conda environment

```bash
git clone https://github.com/zavolanlab/zavolab_pyutils.git
cd zavolab_pyutils
conda env create --file=environment.yml
conda activate zavolab_pyutils
make setup-dev
```
Conda manages non-python dependencies (e.g. bedtools) that are used in several modules of the package.

### From PyPI (non-python dependencies will not be installed)
```bash
pip install zavolab_pyutils
```

### From bioconda (TO DO)
```bash
conda install -c bioconda zavolab_pyutils
```

## Package Structure

The source code is organized into three subpackages under `src/zavolab_pyutils/`:

```
zavolab_pyutils/
├── pySanity/          # Bayesian Sanity normalization (Breda et al. 2021) and fracSanity
│   ├── sanity.py      # apply_sanity_normalization, apply_sanity_normalization_full_bayesian,
│   │                  # test_differential_expression, test_differential_relative_usage,
│   │                  # prepare_isoform_sanity_matrix
│   ├── frac_sanity.py # fit_frac_sanity_map, calculate_differential_recruitment
│   └── visualization.py  # Sanity-specific plots (CI expression, relative usage, v_g, fracSanity)
├── CLI/               # Command-line tools used in Snakemake/Nextflow pipelines
│   ├── motif_metaplot.py          # plot_cs_motifs
│   ├── normalize_umi_lengths.py   # normalize_umi_lengths
│   ├── redefine_qual_and_NHtag.py # redefine_qual_and_NHtag
│   └── transcript_to_genomic.py   # transcript_to_genomic
└── general/           # General-purpose genomic data analysis utilities
    ├── read_count_data_analysis.py  # apply_deseq2_normalization, model_mean_variance, ...
    ├── annotation.py                # parse_gtf_attributes_into_pd_dataframes, ...
    ├── visualization.py             # pca_plot, plot_size_factors, plot_mean_vs_cv, ...
    ├── read_count_simulation.py     # simulate_isoform_poisson_lognormal_counts, ...
    └── parsing_workflow_outputs.py  # parse_mapping_stats, ...
```

All public symbols are also accessible directly from the top-level package:
```python
# via subpackage (explicit)
from zavolab_pyutils.pySanity.sanity import apply_sanity_normalization_full_bayesian
from zavolab_pyutils.general.read_count_data_analysis import apply_deseq2_normalization

# or from the top-level (convenient)
from zavolab_pyutils import apply_sanity_normalization_full_bayesian, apply_deseq2_normalization
```

## Documentation and examples of usage

For various examples of usage and testing, use [test_module.ipynb](test_module.ipynb).

Use AI to ask about the functionality. See the example with Gemini in [docs](docs/).

For further practical examples, please look into other projects where the functions from the package have been used:
[APA localization](https://github.com/zavolanlab/APA_localization/tree/nfya)
this list will be continuosly updated...

## Testing

For various examples of usage and testing, run [test_module.ipynb](test_module.ipynb).

Automatic tests are implemented in [CI workflow](.github/workflows/python-app.yml)

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
