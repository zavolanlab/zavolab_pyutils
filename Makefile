.PHONY: setup-dev

setup-dev:
	@echo "Installing development dependencies..."
	pip install -e .[dev]
	@echo "Setting up Jupyter Notebook stripout..."
	nbstripout --install
	@echo "Development environment setup complete!"
