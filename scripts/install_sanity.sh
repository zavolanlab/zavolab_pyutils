#!/usr/bin/env bash
# Build the C++ Sanity binary (Breda et al., Nat Biotechnol 2021) from GitHub and install it.
#
# Used by `make install-sanity`. Settings (environment variables):
#   SANITY_REF        git branch, tag or commit to build (default: master, i.e. the latest version)
#   SANITY_PREFIX     install prefix; the binary goes to $SANITY_PREFIX/bin/Sanity
#                     (default: $CONDA_PREFIX if a conda env is active, otherwise $HOME/.local)
#   SANITY_ARCH_FLAG  compiler architecture flag (default: upstream's -march=native). On a cluster
#                     whose nodes have different CPUs, use a portable flag, e.g. -march=x86-64-v2
#   SANITY_JOBS       parallel make jobs (default: 4)
#   SANITY_REPO       repository URL (default: https://github.com/jmbreda/Sanity.git)
#   CXX               C++ compiler (set automatically by conda's cxx-compiler package)
set -euo pipefail

SANITY_REPO=${SANITY_REPO:-https://github.com/jmbreda/Sanity.git}
SANITY_REF=${SANITY_REF:-master}
SANITY_PREFIX=${SANITY_PREFIX:-${CONDA_PREFIX:-$HOME/.local}}
SANITY_ARCH_FLAG=${SANITY_ARCH_FLAG:-}
SANITY_JOBS=${SANITY_JOBS:-4}

build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

echo "Cloning $SANITY_REPO ($SANITY_REF)..."
git clone --quiet "$SANITY_REPO" "$build_dir/Sanity"
git -C "$build_dir/Sanity" checkout --quiet "$SANITY_REF"
commit=$(git -C "$build_dir/Sanity" rev-parse --short HEAD)

# Variables given on the make command line override the upstream Makefile's assignments.
make_args=()
if [[ -n ${CXX:-} ]]; then
    make_args+=("CC=$CXX")
fi
if [[ -n $SANITY_ARCH_FLAG ]]; then
    make_args+=("ARCH_FLAG=$SANITY_ARCH_FLAG")
fi

# Inside a conda env, compile and link against the env's zlib and OpenMP runtime, and record
# the env's lib folder as the binary's run path.
if [[ -n ${CONDA_PREFIX:-} ]]; then
    export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
    export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_RUN_PATH="$CONDA_PREFIX/lib${LD_RUN_PATH:+:$LD_RUN_PATH}"
fi

echo "Building Sanity at commit $commit..."
make -C "$build_dir/Sanity/src" -j "$SANITY_JOBS" "${make_args[@]}" Sanity

install -d "$SANITY_PREFIX/bin"
install -m 755 "$build_dir/Sanity/bin/Sanity" "$SANITY_PREFIX/bin/Sanity"

echo "Installed $("$SANITY_PREFIX/bin/Sanity" --version) (commit $commit) to $SANITY_PREFIX/bin/Sanity"
if ! command -v Sanity >/dev/null 2>&1 || [[ $(command -v Sanity) != "$SANITY_PREFIX/bin/Sanity" ]]; then
    echo "Note: $SANITY_PREFIX/bin is not first on PATH for 'Sanity'. Either add it, or set"
    echo "      export SANITY_BIN=$SANITY_PREFIX/bin/Sanity"
fi
