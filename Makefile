.PHONY: help install install-dev test lint format clean docs

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the package in development mode
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e .[dev]

test: ## Run tests
	pytest tests/ -v

lint: ## Run linting checks
	flake8 src/ tests/
	mypy src/

format: ## Format code with black
	black src/ tests/

format-check: ## Check if code is formatted correctly
	black --check src/ tests/

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

docs: ## Build documentation
	@echo "Documentation building not yet implemented"

check: format-check lint test ## Run all checks

pre-commit: format lint test ## Run pre-commit checks

# Development helpers
setup: install-dev ## Set up development environment
	@echo "Development environment set up successfully!"

run-example: ## Run example training (requires data)
	@echo "Example command:"
	@echo "python src/main.py --record_path /path/to/data --record_dir train --save_path ./output --test_path /path/to/test --checkpoint_path ./checkpoints --config_file config/config_germinal.yaml --model_name attention" 