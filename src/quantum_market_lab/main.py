"""Run the first GBM experiment and visualize the simulated paths."""

import numpy as np
import matplotlib.pyplot as plt

from .gbm import simulate_gbm


def main() -> None:
    initial_price = 100.0
    drift = 0.08
    volatility = 0.20
    years = 1.0
    steps = 252
    paths = 10_000

    times, prices = simulate_gbm(
        initial_price=initial_price,
        drift=drift,
        volatility=volatility,
        years=years,
        steps=steps,
        paths=paths,
        seed=42,
    )

    # GBM's theoretical expected price at time T is E[S_T] = S_0 * exp(mu*T).
    theoretical_mean = initial_price * np.exp(drift * years)
    simulated_mean = prices[:, -1].mean()
    standard_error = prices[:, -1].std(ddof=1) / np.sqrt(paths)

    print("=== GBM Phase 1 experiment ===")
    print(f"Paths: {paths:,}")
    print(f"Time steps: {steps}")
    print(f"Theoretical E[S_T]: ${theoretical_mean:.2f}")
    print(f"Monte Carlo mean:    ${simulated_mean:.2f}")
    print(f"Monte Carlo SE:      ${standard_error:.2f}")
    print(
        "Difference in standard errors: "
        f"{abs(simulated_mean - theoretical_mean) / standard_error:.2f}"
    )

    # Plot a manageable subset; all 10,000 paths remain in the simulation.
    fig, ax = plt.subplots(figsize=(11, 6))
    for path in prices[:100]:
        ax.plot(times, path, linewidth=0.7, alpha=0.45)

    ax.set(
        title="Geometric Brownian Motion: 100 Sample Paths",
        xlabel="Time (years)",
        ylabel="Simulated price ($)",
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
