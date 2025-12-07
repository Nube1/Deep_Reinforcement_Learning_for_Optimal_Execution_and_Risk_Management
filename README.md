# Robust Rough Mean Field Games in Carbon Markets

This repository contains a numerical implementation and demonstration of **Robust Rough Mean Field Games (MFG)** applied to carbon emission markets (Cap-and-Trade systems).

The project synthesizes three advanced mathematical frameworks:
1.  **Rough Volatility:** Modeling asset prices with fractional Brownian motion ($H < 0.5$) to capture the "rough" behavior observed in financial time series.
2.  **Mean Field Games:** Modeling the interactions of a large number of agents (firms) optimizing abatement and trading strategies.
3.  **Robust Control:** Introducing "Ambiguity Aversion" ($\eta$), where agents make decisions under model uncertainty.

## Key Features

*   **Rough Heston Simulation:** Comparison of standard volatility ($H=0.5$) vs. rough volatility ($H=0.14$) using Cholesky decomposition.
*   **Markovian Lift:** Implementation of the Abi Jaber & El Euch (2019) method to approximate the fractional kernel with a sum of exponentials, making the non-Markovian system tractable.
*   **Deep Galerkin Method (DGM):** A PyTorch-based Deep Learning solver for the high-dimensional **Hamilton-Jacobi-Bellman-Isaacs (HJBI)** Partial Differential Equation (PDE).
*   **Ambiguity Analysis:** Numerical demonstration of the "Ambiguity Premium," showing how uncertainty drives earlier decarbonization.

## Prerequisites

The code requires Python 3.8+ and the following libraries:

*   **NumPy:** Matrix operations.
*   **PyTorch:** Neural networks and automatic differentiation for DGM.
*   **Matplotlib:** Visualization of paths and control surfaces.
*   **SciPy:** Optimization for the Markovian lift and statistical functions.
*   **Tqdm:** Progress bars for training loops.

### Installation

You can install the required dependencies using pip:

```bash
pip install numpy torch matplotlib scipy tqdm
```

*Note: A GPU (CUDA) is recommended for training the DGM network, but the code will automatically fallback to CPU if not available.*

## Project Structure

The codebase is contained in a single executable script (e.g., `main.py`) which includes the following classes:

1.  **`RoughHestonModel`**: Simulates path-dependent volatility.
2.  **`MarkovianLift`**: Optimizes weights ($c_i$) and rates ($x_i$) to approximate the fractional kernel $K(t) = t^{H-1/2}$.
3.  **`DGMNetwork` & `DGMLayer`**: LSTM-like neural network architecture tailored for solving PDEs (Sirignano & Spiliopoulos, 2018).
4.  **`DGMSolver`**: Manages the training loop to minimize the PDE residual loss.

## Usage

Run the main script to execute the full simulation suite:

```bash
python main.py
```

### Execution Flow
1.  **Rough Volatility Demo:** Simulates price paths and compares statistical properties (skewness, kurtosis) against standard geometric Brownian motion.
2.  **Lift Verification:** Verifies the accuracy of the exponential approximation for the fractional kernel.
3.  **Ambiguity Premium Test:** Calculates optimal controls for varying levels of ambiguity aversion ($\eta$).
4.  **DGM Training:** Trains the neural network to solve the robust control problem over the state space $(t, A, E, S, Y)$.
5.  **Visualization:** Generates and saves plots locally.

## Results & Outputs

The script generates the following visualizations (`.png`) and console outputs:

### 1. Generated Plots
*   **`rough_volatility_comparison.png`**: Visual comparison of price paths and volatility autocorrelation. *Rough volatility exhibits much faster decay in autocorrelation.*
*   **`markovian_lift_approximation.png`**: Shows the fit between the true fractional kernel and the Markovian approximation.
*   **`ambiguity_premium.png`**: Illustrates how increasing $\eta$ (uncertainty aversion) leads to higher abatement efforts.
*   **`dgm_training_loss.png`**: Log-scale plot of the PDE residual loss during training.
*   **`optimal_controls.png`**: The resulting optimal abatement ($u^*$) and trading ($\nu^*$) strategies as a function of allowance inventory.

### 2. Key Statistical Findings
Based on the simulation runs:
*   **Skewness:** Rough volatility models produce significantly higher negative skewness in terminal prices compared to standard diffusion models.
*   **Ambiguity Premium:** As ambiguity aversion ($\eta$) increases:
    *   **Abatement ($u^*$):** Increases significantly. Firms abate more to avoid the risk of high future carbon prices.
    *   **Trading ($\nu^*$):** Decreases. Firms become more cautious in the trading market.

## Theoretical Background

The core PDE solved by the DGM network is the **Robust HJBI Equation**:

$$
-\partial_t V - \inf_{u} \sup_{\nu} \left\{ \mathcal{L}^{u,\nu} V + f(t, x, u, \nu) \right\} = 0
$$

Where the volatility process is non-Markovian (Rough Heston). We lift the state space using auxiliary processes $Y_t^{(i)}$ such that:

$$ V_t \approx V_0 + \sum_{i=1}^n c_i Y_t^{(i)} $$

This allows us to solve the problem using standard Markovian control techniques via Deep Learning.

## License

MIT License. See `LICENSE` for more information.
