# Asymptotic Error of Regularizers

This project contains code for the paper [Unveiling low-dimensional patterns induced by convex
non-differentiable regularizers](https://arxiv.org/abs/2405.07677), where we explore the asymptotic error behavior of various regularizers.

## Project Structure

- `code/plots.py`: Visualizations of asymptotic error and pattern recovery for various regularizers.
- `code/solvers_slope.py`: Implements optimization solvers for SLOPE.
- `code/solvers_glasso.py`: Implements optimization solvers for Generalized Lasso.
- `code/solvers_plotters.py`: Utility functions for plotting.
- `code/test.py`: Contains some test cases.
- `requirements.txt`: List of dependencies.

## Setup Instructions

1. **Clone the repository**  
   ```python
   git clone https://github.com/IvanHejny/asymptotic-error-of-regularizers.git
   cd asymptotic-error-of-regularizers

2. **Create a virtual environment**  
   ```python
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install dependencies**  
   ```python
   pip install -r requirements.txt

4. **Run plots**  
   ```python
   python code/plots.py
python code/plots.py 
- Note: To reproduce the simulations from the paper, you need to modify the 'n' variable (e.g., set n = 15000) within the 'code/plots.py' script or a configuration file before running.
   
