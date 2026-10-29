[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19483710.svg)](https://doi.org/10.5281/zenodo.19483710)
![Tests](https://github.com/zavolanlab/zavolab_pyutils/actions/workflows/python-app.yml/badge.svg)
# zavolab_pyutils

Genomic data analysis utilities from the Zavolan Lab. A collection of Python utilities for common bioinformatics tasks including library size normalization, annotation conversion, and other genomic data analysis operations.

## Features

- **Library Size Normalization**: [Deseq2-like](https://pubmed.ncbi.nlm.nih.gov/25516281/) normalization, [Sanity-like](https://pubmed.ncbi.nlm.nih.gov/33927416/) normalization, termed **pySanity**
- **Sanity_tools**: Python wrappers around the original C++ [Sanity](https://github.com/jmbreda/Sanity) (version >= 2.0) that provide the pySanity outputs (log2 CPM-like expression, condition means and errors, empirical Bayes priors on gene variances, fracSanity), with the core fit done by the C++ binary
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
make install-sanity
```
Conda manages non-python dependencies (e.g. bedtools) that are used in several modules of the package.

`make install-sanity` clones the latest [Sanity](https://github.com/jmbreda/Sanity) from GitHub, compiles it with the env's C++ compiler and installs the binary as `$CONDA_PREFIX/bin/Sanity`; `Sanity_tools` needs it (version >= 2.0). Options, given as environment variables (see [scripts/install_sanity.sh](scripts/install_sanity.sh)):

- `SANITY_REF`: branch, tag or commit to build (default `master`, e.g. `SANITY_REF=v2.0.0`)
- `SANITY_PREFIX`: install prefix (default: the active conda env, otherwise `~/.local`)
- `SANITY_ARCH_FLAG`: compiler architecture flag (default upstream's `-march=native`). On an HPC cluster whose nodes have different CPUs, build a portable binary, e.g. `make install-sanity SANITY_ARCH_FLAG=-march=x86-64-v2`, otherwise jobs on other nodes may fail with "Illegal instruction".

`Sanity_tools` looks for the binary in `$SANITY_BIN`, then `$CONDA_PREFIX/bin/Sanity`, then `Sanity` on `PATH`.

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
├── Sanity_tools/      # wrappers around the C++ Sanity binary (core fit in C++)
│   ├── runner.py         # run_sanity, read_sanity_output, find_sanity_binary
│   ├── normalization.py  # apply_sanity_normalization -> SanityResult
│   └── frac.py           # run_frac_sanity, prepare_frac_sanity_inputs
│                         # (tests, fracSanity models and plots are re-exported from pySanity)
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

`Sanity_tools` is imported as a subpackage only, because its function names overlap with pySanity's:

```python
from zavolab_pyutils import Sanity_tools as st

res = st.apply_sanity_normalization(counts_df, metadata_df, sample_col="sample", cond_col="condition",
                                    v_method="MARG", n_threads=8)
de = st.test_differential_expression(res.means, res.errors_relative, cond_A="Treatment", cond_B="Control")
```

| pySanity | Sanity_tools |
|:---|:---|
| `apply_sanity_normalization_full_bayesian(...)` | `apply_sanity_normalization(..., v_method="MARG")`; `res.to_tuple()` returns the same 7-tuple |
| `apply_sanity_normalization(...)` (point estimate of v_g) | `apply_sanity_normalization(..., v_method="MLE")` (or `"MAP"`, `"EAP"`) |
| `empirical_bayes=True` | `empirical_bayes=True` (v_g fixed at its most probable value under the trend prior, instead of averaged over it) |
| `n_cores` | `n_threads` |
| `vg_df["MAP_v_g"]`, `vg_df["marginalized_v_g"]` | `res.vg["v_g_MLE"]` (pySanity's grid maximum), `res.vg["v_g_posterior_mean"]`; `res.vg["v_g"]` is the value Sanity used |
| fracSanity on means and squared relative errors of two pySanity runs | `run_frac_sanity(counts_df, metadata_df, conditions=("UT", "Stress"), version="v2")`, or `prepare_frac_sanity_inputs(...)` |

Sanity 2.0 fits each gene without the pseudo-count of Sanity 1.x and pySanity (uniform prior on the mean log expression instead of on its fraction), so estimates for genes with only a few counts differ from theirs.

## Documentation and examples of usage

For various examples of usage and testing, use [test_module.ipynb](test_module.ipynb).

Use AI to ask about the functionality. See the example with Gemini in [docs](docs/).

For further practical examples, please look into other projects where the functions from the package have been used:

[NFYA CPA isoforms](https://github.com/zavolanlab/APA_localization/tree/nfya)

[CD47 CPA isoforms](https://github.com/zavolanlab/APA_localization/tree/cd47)

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
