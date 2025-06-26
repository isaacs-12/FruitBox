# FruitBox - AI-Powered Grid Puzzle Solver

FruitBoxAI is an intelligent automation system that solves grid-based puzzles by combining computer vision, machine learning, and automated clicking. The project specifically targets games where players must find rectangles of digits that sum to 10 (e.g. in the original [FruitBox game](https://en.gamesaien.com/game/fruit_box/)). Although this is most interestingly an AI application, it started as a more heuristic solution. The OCR is optimized for speed (on the order of ms for parsing an entire board), and the heuristic method still performs exceptionally well (over ~600 games, it peaked at a score of 135). The AI models are significantly better, consistently scoring in the high 130s to low 140s, with much higher potential.

## 🎯 Project Overview

This project automates the solving of the FruitBox game puzzles where:
- A grid contains digits 1-9
- Players must find rectangles where the sum of digits equals 10
- The goal is to clear as many digits as possible
- The system uses AI to learn optimal solving strategies

Example board:
![image](https://github.com/user-attachments/assets/9750e323-de96-4bf6-abee-af6033f9763c)

## 🏗️ Architecture

The project consists of several key components:

### Core Components

- **`main.py`** - Main entry point that orchestrates the entire process with performance tracking
- **`ocr_grid.py`** - Optical Character Recognition for digit extraction from screenshots
- **`solver.py`** - Multiple solving algorithms (greedy, backtracking, ML-based)
- **`clicker.py`** - Automated mouse clicking to execute solutions
- **`training.py`** - Reinforcement learning environment and training pipeline
- **`gemini.py`** - Alternative OCR implementation using Tesseract
- **`evaluate_model.py`** - Model evaluation on test grids
- **`organize_data.py`** - Data organization and train/validation/test splitting
- **`list_performance.py`** - Performance tracking and visualization

### Supporting Files

- **`utils.py`** - Utility functions for verbose output
- **`eval_comparison.py`** - Evaluation mode comparison and analysis
- **`training_data.py`** - Data collection and management for training
- **`requirements.txt`** - Python dependencies
- **`Makefile`** - Build and automation commands

## 🚀 Features

### 1. Computer Vision & OCR
- **Template Matching**: Uses pre-trained digit templates for accurate recognition
- **Tesseract OCR**: Alternative OCR method for digit extraction
- **Pixel Analysis**: Direct pixel-based digit recognition
- **Grid Detection**: Automatically detects and processes grid boundaries

### 2. Multiple Solving Algorithms
- **Greedy Algorithm**: Fast heuristic approach prioritizing larger rectangles
- **Backtracking**: Exhaustive search for optimal solutions
- **Machine Learning**: Reinforcement learning model trained on thousands of grids
- **Digit Priority**: Optimized algorithms that prioritize clearing more digits

### 3. Automated Execution
- **Screen Capture**: Automatic screenshot capture of game window
- **Coordinate Mapping**: Precise mapping of grid coordinates to screen positions
- **Mouse Automation**: Automated clicking and dragging to draw rectangles
- **Safety Features**: Failsafe mechanisms and coordinate validation

### 4. Machine Learning Training
- **Custom Environment**: Gymnasium-based RL environment for rectangle solving
- **PPO Algorithm**: Proximal Policy Optimization for training
- **Data Collection**: Automated collection of training grids from gameplay
- **Model Evaluation**: Comprehensive evaluation and comparison tools
- **Performance Tracking**: Per-model performance tracking and visualization

## 📁 Project Structure

```
FruitBox/
├── main.py                 # Main entry point with performance tracking
├── ocr_grid.py            # OCR and digit recognition
├── solver.py              # Solving algorithms (greedy, backtracking, ML)
├── clicker.py             # Automated clicking
├── training.py            # ML training environment
├── gemini.py              # Alternative OCR implementation
├── evaluate_model.py      # Model evaluation on test grids
├── organize_data.py       # Data organization and splitting
├── list_performance.py    # Performance tracking and visualization
├── utils.py               # Utility functions
├── eval_comparison.py     # Evaluation analysis
├── training_data.py       # Data collection
├── requirements.txt       # Dependencies
├── Makefile              # Build commands
├── digits/               # Digit template images
├── models/               # Trained ML models
├── screenshots/          # Captured game screenshots
├── training_grids/       # Raw collected training data
├── data/                 # Organized data with train/validation/test split
│   ├── train/           # Training grids (80% of data)
│   ├── validation/      # Validation grids (10% of data)
│   ├── test/            # Test grids (10% of data)
│   └── split_summary.pkl # Data split information
├── training_data/        # Training episode data
├── logs/                 # Training logs
├── plots/               # Training performance plots
├── test_data/           # Test images
└── performance_data_*.json # Per-model performance tracking
```

## 🧠 Model Training Approach

### Reinforcement Learning Environment

The system uses a custom Gymnasium environment that simulates the FruitBox game:

**State Space**: 10×17 grid with digits 0-9
- Each cell contains a digit (0-9)
- Grid represents the current game state
- Zeros indicate cleared cells

**Action Space**: All possible rectangles
- Discrete action space representing rectangle coordinates
- Actions are (r1, c1, r2, c2) defining rectangle corners
- Invalid actions (rectangles that don't sum to 10) are masked out

**Reward Function**: Optimized for digit clearing
```
Reward = non_zero_digits_cleared + completion_bonus
```
- **non_zero_digits_cleared**: Number of non-zero digits in the rectangle (max 10 per action)
- **completion_bonus**: +1000 if grid is completely cleared
- **Efficiency bonus**: Higher rewards for clearing more digits per action

### Training Process

1. **Data Collection**: 
   - Automatically captures grids during gameplay
   - Validates grids have exactly 170 non-zero digits
   - Organizes data into train/validation/test splits

2. **Environment Setup**:
   - Custom Gymnasium environment for rectangle solving
   - Action masking to prevent invalid moves
   - Episode termination when no valid moves remain

3. **Model Training**:
   - **Algorithm**: Proximal Policy Optimization (PPO)
   - **Policy**: Multi-layer perceptron (MLP) with 256×256 hidden layers
   - **Optimization**: Adam optimizer with learning rate 3e-4
   - **Batch Size**: 256 with 2048 steps per update

4. **Evaluation Strategy**:
   - Regular evaluation on validation grids every 30k steps
   - Fast evaluation mode with reduced episodes for speed
   - Model saving based on validation performance

5. **Hardware Optimization**:
   - **Apple Silicon**: Uses MPS (Metal Performance Shaders) for acceleration
   - **CUDA**: Supports NVIDIA GPU acceleration
   - **CPU Fallback**: Optimized CPU training for other systems

### Training Commands

```bash
# Quick training (100k steps)
make train-fast-quick

# Standard training (500k steps)
make train-fast

# Long training (2M steps)
make train-long

# Training with TensorBoard logging
make train-fast-tensorboard

# Force CPU usage
make train-fast-cpu
```

### Model Performance

- **Training Data**: 1,000+ grids with proper train/validation/test split
- **Validation Performance**: Models typically achieve 130-150 digits cleared
- **Training Time**: 30-60 minutes for 500k steps on Apple M4
- **Model Size**: ~2-3MB compressed models

## 🔍 Brute Force Approach

### Algorithm Overview

The brute force approach uses exhaustive search to find the optimal solution by exploring all possible rectangle combinations:

**Core Strategy**:
1. **Generate all valid rectangles**: Find all rectangles where digits sum to 10
2. **Exhaustive search**: Try all possible combinations of rectangles
3. **Optimization**: Find the combination that clears the most digits

### Implementation Details

**Rectangle Generation**:
```python
def find_rectangles_zero_inclusive_greedy(grid):
    # Find all rectangles where non-zero digits sum to 10
    # Include zeros in rectangle size but not in sum
    # Return rectangles sorted by digit clearing efficiency
```

**Search Algorithms**:

1. **Greedy Approach** (`find_rectangles_digit_priority_greedy`):
   - Prioritizes rectangles that clear more digits
   - Fast but not guaranteed optimal
   - Time complexity: O(n²) where n = grid cells

2. **Backtracking Search** (`solve_rectangles_backtracking`):
   - Exhaustive search of all rectangle combinations
   - Guaranteed to find optimal solution
   - Time complexity: O(2^n) - exponential growth

3. **Digit Priority Greedy**:
   - Optimized greedy algorithm
   - Balances speed and solution quality
   - Time complexity: O(n² log n)

### Performance Characteristics

**Greedy Algorithm**:
- **Speed**: Very fast (< 1 second)
- **Quality**: Good solutions (120-140 digits cleared)
- **Guarantee**: Not optimal but practical

**Backtracking Algorithm**:
- **Speed**: Slow (minutes to hours for complex grids)
- **Quality**: Optimal solution
- **Guarantee**: Always finds best possible solution

**Digit Priority Greedy**:
- **Speed**: Fast (1-5 seconds)
- **Quality**: Very good solutions (130-150 digits cleared)
- **Guarantee**: Not optimal but excellent practical performance

### Use Cases

**Greedy**: Real-time solving during gameplay
**Backtracking**: Analysis and verification of optimal solutions
**Digit Priority**: Best balance of speed and quality for most applications

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- Tesseract OCR (for OCR functionality)
- Virtual environment (recommended)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd FruitBox
   ```

2. **Create virtual environment:**
   ```bash
   make venv-create
   ```

3. **Activate virtual environment:**
   ```bash
   source fruitbox_env/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Basic Usage

1. **Run the main solver:**
   ```bash
   make run
   ```

2. **Run with verbose output:**
   ```bash
   make run-verbose
   ```

3. **Test with sample image:**
   ```bash
   make run-test
   ```

4. **Run with specific model:**
   ```bash
   make run-model MODEL=models/model_20250626_102548.zip
   ```

### Data Organization

1. **Organize training data:**
   ```bash
   make organize-data
   ```

2. **Verify data isolation:**
   ```bash
   make organize-data-verify
   ```

### Machine Learning Training

1. **Quick training:**
   ```bash
   make train-fast
   ```

2. **Long training:**
   ```bash
   make train-long
   ```

3. **Training with TensorBoard:**
   ```bash
   make train-fast-tensorboard
   ```

4. **Force CPU training:**
   ```bash
   make train-fast-cpu
   ```

### Model Evaluation

1. **Evaluate latest model:**
   ```bash
   make evaluate-latest
   ```

2. **Evaluate with multiple episodes:**
   ```bash
   make evaluate-latest-many
   ```

3. **List available models:**
   ```bash
   make list-models
   ```

4. **List performance data:**
   ```bash
   make list-performance
   ```

5. **Continue training from existing model:**
   ```bash
   make continue-train MODEL=models/your_model.zip
   ```

## 🔧 Configuration

### Screen Coordinates
Adjust the screen coordinates in `main.py` for your specific setup:
```python
region = (18, 202, 1223, 723)  # (left, top, width, height)
screen_offset = (18, 202)     # (x, y) offset
```

and also the coordinates of the start and reset buttons to enable auto-play:
```python
RESET_LOC = (55,1011)
PLAY_LOC = (320,613)
```

### Solving Methods
Choose from different solving approaches:
- `'template'` - Template matching for digit recognition
- `'ocr'` - Tesseract OCR
- `'pixel'` - Direct pixel analysis

### Training Parameters
Configure training in `training.py`:
- `total_timesteps` - Total training steps
- `eval_freq` - Evaluation frequency
- `eval_episodes` - Episodes per evaluation

## 📊 Performance

The system achieves:
- **Digit Recognition**: 95%+ accuracy using template matching
- **Solving Speed**: Sub-second solving for most grids
- **Clicking Accuracy**: Precise coordinate mapping and execution
- **Training Efficiency**: Fast convergence with PPO algorithm
- **Model Performance**: 130-150 digits cleared consistently

## 🔍 Debugging

### Verbose Mode
Enable detailed output for debugging:
```bash
python main.py --verbose
```

### Debug Images
The system saves debug images during processing:
- `preview_overlay.png` - Solution visualization
- `debug_steps/` - Intermediate processing steps
- `debug_samples/` - Individual digit recognition samples

### Logs
Training logs are saved in the `logs/` directory with detailed metrics.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenCV** for computer vision capabilities
- **Tesseract** for OCR functionality
- **PyAutoGUI** for automation
- **Stable Baselines3** for reinforcement learning
- **Gymnasium** for the RL environment framework
- **PyTorch** for deep learning framework

## 🐛 Known Issues

- Screen coordinates may need adjustment for different screen resolutions
- OCR accuracy can vary based on font and image quality
- Training requires significant computational resources (though it will take advantage of your hardware if you have Apple Silicon, NVidia Cuda, or just a CPU)
- Brute force backtracking can be very slow for complex grids
