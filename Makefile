.PHONY: setup-dev

setup-dev:
	@echo "Installing development dependencies..."
	pip install -e .[dev]
	@echo "Development environment setup complete!"
