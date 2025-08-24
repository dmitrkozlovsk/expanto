# Expanto Makefile - Simple git clone workflow

.PHONY: help setup run start stop clean

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m

help: ## Show available commands
	@echo "$(GREEN)Expanto - A/B Testing Platform$(NC)"
	@echo ""
	@echo "$(YELLOW)Quick start:$(NC)"
	@echo "  make setup    # Setup project with examples"
	@echo "  make start    # Start both Streamlit + AI assistant"
	@echo "  make stop     # Stop all services"
	@echo ""

setup: ## Setup project with examples and database
	@echo "$(GREEN)🚀 Setting up Expanto...$(NC)"
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "$(YELLOW)Installing uv...$(NC)"; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@echo "Installing dependencies..."
	@uv sync
	@echo "Creating directories..."
	@mkdir -p .streamlit
	@echo "Copying configs..."
	@for file in demo_and_setup/configs/*; do \
		basename_file=$$(basename "$$file"); \
		if [ ! -f ".streamlit/$$basename_file" ]; then \
			echo "  Copying $$basename_file..."; \
			cp "$$file" ".streamlit/$$basename_file"; \
		else \
			echo "  $$basename_file already exists, skipping..."; \
		fi; \
	done
	@echo "Setting up database..."
	@uv run python -m demo_and_setup.first_experiment
	@echo "$(GREEN)✅ Setup complete!$(NC)"
	@echo "Edit .streamlit/secrets.toml with your credentials"

run: ## Start Streamlit only
	@uv run streamlit run app.py

start: ## Start both Streamlit and AI assistant
	@echo "$(GREEN)🚀 Starting Expanto...$(NC)"
	@uv run fastapi dev assistant/app.py &
	@sleep 2
	@uv run streamlit run app.py

stop: ## Stop all services
	@pkill -f "streamlit run app.py" || echo "Streamlit not running"
	@pkill -f "python assistant/app.py" || echo "Assistant not running"

clean: ## Clean cache files
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true

# Default shows help
default: help