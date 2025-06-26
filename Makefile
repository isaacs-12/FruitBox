# Run the main script
run:
	python main.py

# Run the main script with verbose output
run-verbose:
	python main.py --verbose

# Run the main script with test image
run-test:
	python main.py --test

# Run the main script with a specific model
run-model:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: Please specify a model path using MODEL=path/to/model.zip"; \
		echo "Example: make run-model MODEL=models/model_20250626_102548.zip"; \
		exit 1; \
	fi
	python main.py --model $(MODEL)

# Run the main script with a specific model and verbose output
run-model-verbose:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: Please specify a model path using MODEL=path/to/model.zip"; \
		echo "Example: make run-model-verbose MODEL=models/model_20250626_102548.zip"; \
		exit 1; \
	fi
	python main.py --model $(MODEL) --verbose

# Virtual environment management
venv-create:
	python3 -m venv fruitbox_env
	. fruitbox_env/bin/activate && pip install --upgrade pip
	. fruitbox_env/bin/activate && pip install -r requirements.txt
	. fruitbox_env/bin/activate && pip install tensorboard

venv-activate:
	@echo "To activate the virtual environment, run:"
	@echo "source fruitbox_env/bin/activate"

# Data organization (IMPORTANT: Run this first to prevent data contamination!)
organize-data:
	. fruitbox_env/bin/activate && python organize_data.py

organize-data-verify:
	. fruitbox_env/bin/activate && python organize_data.py --verify

# Training commands (IMPORTANT: Run 'make organize-data' first!)
train:
	. fruitbox_env/bin/activate && python training.py --mode train

train-long:
	. fruitbox_env/bin/activate && python training.py --mode train --timesteps 2000000

train-fast:
	. fruitbox_env/bin/activate && python training.py --mode train --timesteps 500000 --use-saved-grids

train-fast-cpu:
	. fruitbox_env/bin/activate && python training.py --mode train --timesteps 500000 --use-saved-grids --cpu

train-fast-quick:
	. fruitbox_env/bin/activate && python training.py --mode train --timesteps 100000 --use-saved-grids --fast --fast-eval

train-fast-quick-cpu:
	. fruitbox_env/bin/activate && python training.py --mode train --timesteps 100000 --use-saved-grids --fast --fast-eval --cpu

train-fast-tensorboard:
	. fruitbox_env/bin/activate && python training.py --mode train --timesteps 500000 --use-saved-grids --tensorboard

train-fast-tensorboard-cpu:
	. fruitbox_env/bin/activate && python training.py --mode train --timesteps 500000 --use-saved-grids --tensorboard --cpu

train-clean: venv-create train

# Model evaluation
evaluate:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: Please specify a model path using MODEL=path/to/model.zip"; \
		exit 1; \
	fi
	. fruitbox_env/bin/activate && python training.py --mode evaluate --model-path $(MODEL)

evaluate-many:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: Please specify a model path using MODEL=path/to/model.zip"; \
		exit 1; \
	fi
	. fruitbox_env/bin/activate && python training.py --mode evaluate --model-path $(MODEL) --eval-episodes 10

# Evaluate latest model on test data (proper evaluation)
evaluate-latest:
	. fruitbox_env/bin/activate && python evaluate_model.py

evaluate-latest-many:
	. fruitbox_env/bin/activate && python evaluate_model.py --episodes 5

# Continue training from a saved model
continue-train:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: Please specify a model path using MODEL=path/to/model.zip"; \
		exit 1; \
	fi
	. fruitbox_env/bin/activate && python training.py --mode continue --model-path $(MODEL)

continue-train-long:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: Please specify a model path using MODEL=path/to/model.zip"; \
		exit 1; \
	fi
	. fruitbox_env/bin/activate && python training.py --mode continue --model-path $(MODEL) --timesteps 2000000

# List available models
list-models:
	@echo "Available models in models/ directory:"
	@ls -l models/*.zip 2>/dev/null || echo "No models found in models/ directory"

# List performance data for all models
list-performance:
	python list_performance.py

# Clean up generated files
clean:
	rm -rf models/*.zip
	rm -rf __pycache__
	rm -rf fruitbox_env

# Solve a grid using a trained model
solve:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: Please specify a model path using MODEL=path/to/model.zip"; \
		exit 1; \
	fi
	. fruitbox_env/bin/activate && python training.py --mode solve --model-path $(MODEL)

solve-quiet:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: Please specify a model path using MODEL=path/to/model.zip"; \
		exit 1; \
	fi
	. fruitbox_env/bin/activate && python training.py --mode solve --model-path $(MODEL) --no-render

# Create training plots
plot:
	@echo "Creating training performance plots..."
	@echo "Available models:"
	@ls -1 models/*.zip 2>/dev/null | sed 's/models\///' | sed 's/\.zip//' || echo "No models found"
	@echo ""
	@echo "To create a plot for a specific model, use:"
	@echo "make plot-model MODEL=model_timestamp"

plot-model:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: Please specify a model timestamp using MODEL=timestamp"; \
		echo "Example: make plot-model MODEL=20250617_220933"; \
		exit 1; \
	fi
	. fruitbox_env/bin/activate && python training.py --create-plot $(MODEL)

.PHONY: run run-verbose run-test run-model run-model-verbose venv-create venv-activate organize-data organize-data-verify train train-long train-fast train-fast-cpu train-fast-quick train-fast-quick-cpu train-fast-tensorboard train-fast-tensorboard-cpu train-clean evaluate evaluate-many evaluate-latest evaluate-latest-many continue-train continue-train-long list-models list-performance clean solve solve-quiet plot plot-model