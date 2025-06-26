# Run the main script
run:
	python main.py

# Run the main script with verbose output
run-verbose:
	python main.py --verbose

# Run the main script with test image
run-test:
	python main.py --test

# Virtual environment management
venv-create:
	python3.7 -m venv fruitbox_env
	. fruitbox_env/bin/activate && pip install --upgrade pip
	. fruitbox_env/bin/activate && pip install -r requirements.txt
	. fruitbox_env/bin/activate && pip install tensorboard

venv-activate:
	@echo "To activate the virtual environment, run:"
	@echo "source fruitbox_env/bin/activate"

# Training commands
train:
	. fruitbox_env/bin/activate && python training.py --mode train

train-long:
	. fruitbox_env/bin/activate && python training.py --mode train --timesteps 2000000

train-fast:
	. fruitbox_env/bin/activate && python training.py --mode train --timesteps 500000 --use-saved-grids

train-fast-quick:
	. fruitbox_env/bin/activate && python training.py --mode train --timesteps 100000 --use-saved-grids --fast --fast-eval

train-fast-tensorboard:
	. fruitbox_env/bin/activate && python training.py --mode train --timesteps 500000 --use-saved-grids --tensorboard

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

.PHONY: run run-verbose run-test venv-create venv-activate train train-long train-fast train-fast-quick train-fast-tensorboard train-clean evaluate evaluate-many continue-train continue-train-long list-models clean solve solve-quiet