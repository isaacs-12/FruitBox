# FruitBox - AI-Powered Grid Puzzle Solver

FruitBoxAI is an intelligent automation system that solves grid-based puzzles by combining computer vision, machine learning, and automated clicking. The project specifically targets games where players must find rectangles of digits that sum to 10 (e.g. in the original FruitBox game). Although this is most interestingly an AI application, it started as a more heuristic solution. The OCR is optimized for speed (on the order of ms for parsing an entire board), and the heuristic method still performs exceptionally well (over ~600 games, it peaked at a score of 135). The AI models are significantly better, consistently scoring in the high 130s to low 140s, with much higher potential.

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

- **`main.py`** - Main entry point that orchestrates the entire process
- **`ocr_grid.py`** - Optical Character Recognition for digit extraction from screenshots
- **`solver.py`** - Multiple solving algorithms (greedy, backtracking, ML-based)
- **`clicker.py`** - Automated mouse clicking to execute solutions
- **`training.py`** - Reinforcement learning environment and training pipeline
- **`gemini.py`** - Alternative OCR implementation using Tesseract

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

## 📁 Project Structure

```
FruitBox/
├── main.py                 # Main entry point
├── ocr_grid.py            # OCR and digit recognition
├── solver.py              # Solving algorithms
├── clicker.py             # Automated clicking
├── training.py            # ML training environment
├── gemini.py              # Alternative OCR implementation
├── utils.py               # Utility functions
├── eval_comparison.py     # Evaluation analysis
├── training_data.py       # Data collection
├── requirements.txt       # Dependencies
├── Makefile              # Build commands
├── digits/               # Digit template images
├── models/               # Trained ML models
├── screenshots/          # Captured game screenshots
├── training_grids/       # Collected training data
├── training_data/        # Training episode data
├── logs/                 # Training logs
└── test_data/           # Test images
└── data/                 # Grids parsed into train/test/eval for isolation

```

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

### Model Evaluation

1. **Evaluate a model:**
   ```bash
   make evaluate MODEL=models/your_model.zip
   ```

2. **List available models:**
   ```bash
   make list-models
   ```

3. **Continue training from existing model:**
   ```bash
   make continue-train MODEL=models/your_model.zip
   ```

## 🔧 Configuration

### Screen Coordinates
Adjust the screen coordinates in `main.py` for your specific setup:
```python
region = (10, 172, 1431, 842)  # (left, top, width, height)
screen_offset = (10, 172)     # (x, y) offset
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

## 🤖 Machine Learning Details

### Environment
- **State Space**: 10x17 grid with digits 0-9
- **Action Space**: All possible rectangles (discrete)
- **Reward Function**: Based on digits cleared with completion bonuses

### Training Process
1. **Data Collection**: Automatically captures grids during gameplay
2. **Environment Setup**: Custom Gymnasium environment for rectangle solving
3. **Model Training**: PPO algorithm with action masking
4. **Evaluation**: Regular evaluation on validation grids
5. **Model Saving**: Automatic saving of best performing models

### Model Performance
- Trained models can solve grids with 130+ digits cleared
- Evaluation includes multiple metrics and comparison tools
- Supports model continuation and fine-tuning

## 📊 Performance

The system achieves:
- **Digit Recognition**: 95%+ accuracy using template matching
- **Solving Speed**: Sub-second solving for most grids
- **Clicking Accuracy**: Precise coordinate mapping and execution
- **Training Efficiency**: Fast convergence with PPO algorithm

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

## 🐛 Known Issues

- Screen coordinates may need adjustment for different screen resolutions
- OCR accuracy can vary based on font and image quality
- Training requires significant computational resources (though it will take advantage of your hardware if you have Apple Silicon, NVidia Cuda, or just a CPU)
