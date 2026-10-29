"""
Thin wrapper around the C++ Sanity binary (https://github.com/jmbreda/Sanity, version >= 2.0).

Writes a count matrix in Sanity's TSV format, runs Sanity with extended output and reads all
output files back into pandas objects, in Sanity's own natural-log units.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

SANITY_REPO_URL = "https://github.com/jmbreda/Sanity"
MIN_SANITY_VERSION = (2, 0, 0)
V_METHODS = ("MARG", "MAP", "EAP", "MLE")

# Sanity's defaults for the variance grid
DEFAULT_VMIN = 0.001
DEFAULT_VMAX = 50.0
DEFAULT_NBIN = 160

_OUTPUT_FILES = (
    "log_transcription_quotients.txt", "ltq_error_bars.txt", "mu.txt", "d_mu.txt",
    "variance.txt", "delta.txt", "d_delta.txt", "likelihood.txt", "geneID.txt", "cellID.txt",
)


class SanityNotFoundError(RuntimeError):
    """Raised when no suitable Sanity binary (version >= 2.0) can be found."""


def _install_hint():
    return (
        "Install the latest Sanity from GitHub with `make install-sanity` in the zavolab_pyutils "
        f"repository (see README), or build it from {SANITY_REPO_URL} and set the environment "
        "variable SANITY_BIN to the binary."
    )


def _parse_version(text):
    match = re.search(r"Sanity version (\d+)\.(\d+)(?:\.(\d+))?", text)
    if match is None:
        return None
    return tuple(int(x) if x is not None else 0 for x in match.groups())


def _query_version(sanity_bin):
    """(version tuple or None, console output) of `Sanity --version`."""
    try:
        res = subprocess.run([sanity_bin, "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as err:
        return None, str(err)
    output = (res.stdout + res.stderr).strip()
    # Sanity 1.x has no --version option and prints its usage instead
    return (_parse_version(output) if res.returncode == 0 else None), output


def get_sanity_version(sanity_bin: Optional[str] = None) -> Optional[str]:
    """
    Version reported by `Sanity --version`, e.g. '2.0.0'; None if the binary does not report one
    (Sanity < 2.0, or it fails to run).

    Parameters
    ----------
    sanity_bin : str or None, optional
        Path to the binary. None searches as in `find_sanity_binary`.
    """
    sanity_bin = sanity_bin or find_sanity_binary(check_version=False)
    version, _ = _query_version(sanity_bin)
    return None if version is None else ".".join(str(v) for v in version)


def find_sanity_binary(sanity_bin: Optional[str] = None, check_version: bool = True) -> str:
    """
    Locates the Sanity binary.

    Search order: `sanity_bin`, the environment variable SANITY_BIN, $CONDA_PREFIX/bin/Sanity,
    then `Sanity` on PATH.

    Parameters
    ----------
    sanity_bin : str or None, optional
        Explicit path to the binary.
    check_version : bool, optional
        Require Sanity >= 2.0, whose command-line interface this module uses. Default is True.

    Returns
    -------
    str
        Absolute path to the binary.
    """
    candidates = [sanity_bin, os.environ.get("SANITY_BIN")]
    if os.environ.get("CONDA_PREFIX"):
        candidates.append(os.path.join(os.environ["CONDA_PREFIX"], "bin", "Sanity"))
    candidates.append(shutil.which("Sanity"))

    for cand in candidates:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            path = os.path.abspath(cand)
            break
    else:
        raise SanityNotFoundError("Sanity binary not found. " + _install_hint())

    if check_version:
        version, output = _query_version(path)
        if version is None or version < MIN_SANITY_VERSION:
            last_line = output.splitlines()[-1] if output else ""
            raise SanityNotFoundError(
                f"{path} is not a working Sanity >= 2.0 (`--version` gave: {last_line!r}). "
                "Sanity_tools needs version 2.0 or later (the -v_m option replaced -max_v). "
                + _install_hint()
            )
    return path


@dataclass
class SanityOutput:
    """
    Raw output of one Sanity run, in Sanity's natural-log units. Gene-by-cell tables have genes
    as rows and cells (samples) as columns.

    Attributes
    ----------
    ltq : pd.DataFrame
        log_transcription_quotients.txt: log transcription quotients x_gc = mu_g + delta_gc.
    ltq_error : pd.DataFrame
        ltq_error_bars.txt: error bar (s.d.) of x_gc, including the uncertainty of mu_g.
    mu, d_mu : pd.Series
        mean LTQ of each gene and its error bar (s.d.).
    variance : pd.Series
        v_g as reported for the chosen v_method (MARG: posterior mean; MAP/MLE/EAP: the grid value
        used for the estimates).
    delta, d_delta : pd.DataFrame
        log fold changes delta_gc = x_gc - mu_g and their error bars (s.d.).
    likelihood : pd.DataFrame
        normalised likelihood of v_g on the variance grid (genes x grid values). It contains no
        prior: Sanity's MAP divides it by v (a 1/v prior) when choosing v_g.
    v_grid : np.ndarray
        variance grid, reconstructed from vmin, vmax and nbin (Sanity prints six decimals, so the
        header of likelihood.txt is zero for grid values below 5e-7).
    v_method, version, command : str
        Taken from sanity_command.txt.
    output_dir : Path or None
        Folder of the output files, None if they were in a temporary folder.
    log : str
        Sanity's console output.
    """

    ltq: pd.DataFrame
    ltq_error: pd.DataFrame
    mu: pd.Series
    d_mu: pd.Series
    variance: pd.Series
    delta: pd.DataFrame
    d_delta: pd.DataFrame
    likelihood: pd.DataFrame
    v_grid: np.ndarray
    v_method: str
    version: str
    command: str
    output_dir: Optional[Path] = None
    log: str = field(default="", repr=False)


def write_sanity_input(counts_df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """
    Writes a genes x samples count matrix in Sanity's tab-separated input format.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Non-negative counts (integer or fractional), genes as rows, samples as columns.
    path : str or Path
        Output TSV path.
    """
    values = counts_df.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Counts must be finite and non-negative.")
    for label in list(counts_df.index.astype(str)) + list(counts_df.columns.astype(str)):
        if any(ch in label for ch in "\t\n\r"):
            raise ValueError(f"Gene and sample names must not contain tabs or newlines: {label!r}")
    if counts_df.index.has_duplicates or counts_df.columns.has_duplicates:
        raise ValueError("Gene and sample names must be unique.")
    path = Path(path)
    counts_df.to_csv(path, sep="\t", index_label="GeneID")
    return path


def _parse_command(command):
    """vmin, vmax and nbin from the command line recorded in sanity_command.txt."""
    params = {"vmin": DEFAULT_VMIN, "vmax": DEFAULT_VMAX, "nbin": DEFAULT_NBIN}
    flags = {"-vmin": "vmin", "--variance_min": "vmin", "-vmax": "vmax", "--variance_max": "vmax",
             "-nbin": "nbin", "--number_of_variance_bins": "nbin"}
    tokens = shlex.split(command) if command else []
    for tok, val in zip(tokens[:-1], tokens[1:]):
        if tok in flags:
            key = flags[tok]
            params[key] = int(val) if key == "nbin" else float(val)
    return params


def variance_grid(vmin: float, vmax: float, nbin: int) -> np.ndarray:
    """Sanity's variance grid, v_k = vmin * exp(k * log(vmax / vmin) / (nbin - 1))."""
    return vmin * np.exp(np.log(vmax / vmin) / (nbin - 1) * np.arange(nbin))


def read_sanity_output(output_dir: Union[str, Path]) -> SanityOutput:
    """
    Reads the output of a Sanity (>= 2.0) run made with extended output (-e 1).

    Parameters
    ----------
    output_dir : str or Path
        Folder given to Sanity with -d.

    Returns
    -------
    SanityOutput
    """
    d = Path(output_dir)
    missing = [f for f in _OUTPUT_FILES if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{d} lacks Sanity output files {missing}. Sanity_tools needs a Sanity >= 2.0 run with "
            "extended output (-e 1); gzip and NPY outputs are not supported."
        )

    meta = {"version": "unknown", "v_method": "unknown", "command": ""}
    cmd_file = d / "sanity_command.txt"
    if cmd_file.exists():
        for line in cmd_file.read_text().splitlines():
            if line.startswith("# Sanity version:"):
                meta["version"] = line.split(":", 1)[1].strip()
            elif line.startswith("# Method:"):
                meta["v_method"] = line.split(":", 1)[1].strip()
            elif line.strip() and not line.startswith("#"):
                meta["command"] = line.strip()

    genes = pd.Index((d / "geneID.txt").read_text().splitlines(), name="GeneID")
    cells = pd.Index((d / "cellID.txt").read_text().splitlines())

    def matrix(name, header):
        if header:
            df = pd.read_csv(d / name, sep="\t", dtype={"GeneID": str}).set_index("GeneID")
            values = df.to_numpy(dtype=float)
        else:
            values = pd.read_csv(d / name, sep="\t", header=None).to_numpy(dtype=float)
        if values.shape != (len(genes), len(cells)):
            raise ValueError(f"{name}: shape {values.shape} does not match geneID/cellID.")
        return pd.DataFrame(values, index=genes, columns=cells)

    def vector(name):
        values = pd.read_csv(d / name, header=None)[0].to_numpy(dtype=float)
        if len(values) != len(genes):
            raise ValueError(f"{name}: {len(values)} values for {len(genes)} genes.")
        return pd.Series(values, index=genes, name=name.removesuffix(".txt"))

    lik = pd.read_csv(d / "likelihood.txt", sep="\t", header=0)
    lik_values = lik.iloc[:, 1:].to_numpy(dtype=float)
    grid = variance_grid(**_parse_command(meta["command"]))
    if len(grid) != lik_values.shape[1]:
        # command line not available or unparseable: fall back to the printed header
        grid = lik.columns[1:].astype(float).to_numpy()
    likelihood = pd.DataFrame(lik_values, index=genes, columns=pd.Index(grid, name="v"))

    return SanityOutput(
        ltq=matrix("log_transcription_quotients.txt", header=True),
        ltq_error=matrix("ltq_error_bars.txt", header=True),
        mu=vector("mu.txt"),
        d_mu=vector("d_mu.txt"),
        variance=vector("variance.txt"),
        delta=matrix("delta.txt", header=False),
        d_delta=matrix("d_delta.txt", header=False),
        likelihood=likelihood,
        v_grid=grid,
        v_method=meta["v_method"],
        version=meta["version"],
        command=meta["command"],
        output_dir=d,
    )


def run_sanity(
    counts_df: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    v_method: str = "MARG",
    vmin: float = DEFAULT_VMIN,
    vmax: float = DEFAULT_VMAX,
    nbin: int = DEFAULT_NBIN,
    n_threads: int = 4,
    no_norm: bool = False,
    overwrite: bool = False,
    sanity_bin: Optional[str] = None,
    verbose: bool = False,
) -> SanityOutput:
    """
    Runs the C++ Sanity binary on a count matrix and reads its output.

    Genes without any counts are dropped by Sanity, so the output may have fewer genes than
    `counts_df`; gene and sample labels and their order are otherwise preserved.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Non-negative counts, genes as rows, samples (cells) as columns.
    output_dir : str, Path or None, optional
        Folder for Sanity's input and output files, kept after the run. None uses a temporary
        folder that is deleted afterwards. Default is None.
    v_method : {'MARG', 'MAP', 'EAP', 'MLE'}, optional
        Sanity's -v_m option. 'MARG' averages the estimates over the posterior of v_g (the
        original Sanity method, as in pySanity's full Bayesian mode); the others fix v_g at the
        maximum a posteriori value under a 1/v prior (MAP, Sanity's default), the grid value
        nearest the posterior mean (EAP) or the maximum-likelihood value (MLE). Default is 'MARG'.
    vmin, vmax, nbin : float, float, int, optional
        Range and number of points of the log-spaced grid of v_g. Defaults are Sanity's
        (0.001, 50, 160).
    n_threads : int, optional
        Sanity's -n option. Default is 4.
    no_norm : bool, optional
        Skip normalisation by total counts per sample (Sanity's -no_norm). Default is False.
    overwrite : bool, optional
        Allow writing into an `output_dir` that already contains Sanity output. Default is False.
    sanity_bin : str or None, optional
        Path to the binary; None searches as in `find_sanity_binary`.
    verbose : bool, optional
        Print Sanity's console output. Default is False.

    Returns
    -------
    SanityOutput
    """
    v_method = v_method.upper()
    if v_method not in V_METHODS:
        raise ValueError(f"v_method must be one of {V_METHODS}.")
    if not (0 < vmin <= vmax) or nbin < 2:
        raise ValueError("Require 0 < vmin <= vmax and nbin >= 2.")
    sanity_bin = find_sanity_binary(sanity_bin)

    tmp = None
    if output_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="sanity_")
        out = Path(tmp.name)
    else:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        existing = [f for f in _OUTPUT_FILES if (out / f).exists()]
        if existing and not overwrite:
            raise FileExistsError(
                f"{out} already contains Sanity output ({existing[0]}, ...). Use another folder "
                "or overwrite=True."
            )
    try:
        input_file = write_sanity_input(counts_df, out / "sanity_input_counts.tsv")
        cmd = [
            sanity_bin, "-f", str(input_file), "-d", str(out), "-e", "1", "-v_m", v_method,
            "-vmin", repr(float(vmin)), "-vmax", repr(float(vmax)), "-nbin", str(int(nbin)),
            "-n", str(int(n_threads)), "-no_norm", "1" if no_norm else "0",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        log = res.stdout + res.stderr
        if verbose:
            print(log)
        if res.returncode != 0:
            raise RuntimeError(
                f"Sanity failed (exit code {res.returncode}):\n{shlex.join(cmd)}\n"
                + "\n".join(log.splitlines()[-20:])
            )
        result = read_sanity_output(out)
        result.log = log
        if tmp is not None:
            result.output_dir = None

        # restore the caller's labels (Sanity keeps the order and drops genes without counts)
        kept = counts_df.index[counts_df.to_numpy(dtype=float).sum(axis=1) > 0]
        if (list(kept.astype(str)) != list(result.mu.index)
                or list(counts_df.columns.astype(str)) != list(result.ltq.columns)):
            raise RuntimeError("Sanity output genes or samples do not match the input matrix.")
        for name in ("ltq", "ltq_error", "delta", "d_delta"):
            df = getattr(result, name)
            df.index, df.columns = kept, counts_df.columns
        for name in ("mu", "d_mu", "variance"):
            getattr(result, name).index = kept
        result.likelihood.index = kept
        return result
    finally:
        if tmp is not None:
            tmp.cleanup()
