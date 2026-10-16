.PHONY: setup-dev install-sanity

setup-dev:
	@echo "Installing development dependencies..."
	pip install -e .[dev]
	@echo "Development environment setup complete!"

# Build the latest C++ Sanity from https://github.com/jmbreda/Sanity and install it into the
# active conda env (or ~/.local). Options: SANITY_REF, SANITY_PREFIX, SANITY_ARCH_FLAG, SANITY_JOBS,
# see scripts/install_sanity.sh
install-sanity:
	bash scripts/install_sanity.sh
