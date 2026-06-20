# Predictive Control Loop Jitter Lab

Python-based robotics runtime validation framework built around Extended Kalman Filter localization. The project benchmarks fixed-rate control-loop timing under stress, measures localization quality, and trains a Random Forest model to predict next-cycle deadline misses from runtime telemetry.

## Overview

Predictive Control Loop Jitter Lab explores a practical robotics systems question: how do timing instability and system load affect a localization loop, and can upcoming deadline misses be predicted before they happen?

The framework combines:

- an EKF localization core adapted from PythonRobotics
- a fixed-rate benchmark loop with deadline enforcement
- runtime telemetry collection for timing and resource metrics
- configurable stress injection scenarios
- visualization utilities for runtime analysis
- a Random Forest model for next-cycle deadline miss prediction

This project is designed as a robotics software engineering portfolio piece with a strong emphasis on real-time behavior, observability, and validation methodology.

## Why This Project Matters

In robotics, a loop that is algorithmically correct can still fail operationally if it runs late, jitters heavily, or misses deadlines under load. This project focuses on that systems layer by connecting timing behavior to algorithm performance and predictive monitoring.

Key engineering questions addressed here:

- How stable is a 100 ms localization loop under nominal and stressed conditions?
- Which stress modes most strongly increase jitter and deadline misses?
- Does degraded runtime behavior immediately translate to worse localization error?
- Which telemetry signals best predict a deadline miss on the next control cycle?

## Architecture

```text
+---------------------------+
| Fixed-Rate Benchmark Loop |
| target period: 100 ms     |
+-------------+-------------+
              |
              v
+---------------------------+        +---------------------------+
| EKF Core                  |        | Stress Injector           |
| state propagation         |<-------| none / random_delay /    |
| measurement update        |        | cpu_stress / mixed        |
+-------------+-------------+        +-------------+-------------+
              |                                      |
              v                                      v
+---------------------------+        +---------------------------+
| Runtime Metrics Logger    |        | EKF Quality Metrics       |
| execution time            |        | position error            |
| loop period               |        | covariance trace          |
| jitter                    |        +---------------------------+
| CPU / memory              |
| deadline misses           |
+-------------+-------------+
              |
              v
+---------------------------+
| CSV Logs + Analysis       |
| plots, benchmark reports, |
| ML training datasets      |
+-------------+-------------+
              |
              v
+---------------------------+
| Random Forest Predictor   |
| X[t] -> deadline_miss[t+1]|
+---------------------------+
```

## Implemented Features

- EKF localization core adapted from PythonRobotics
- Fixed-rate benchmark loop using `time.perf_counter()`
- Runtime metrics logger with CSV export
- Stress injector with four modes:
  - `none`
  - `random_delay`
  - `cpu_stress`
  - `mixed`
- Logged runtime and EKF metrics:
  - `loop_execution_time_ms`
  - `loop_period_ms`
  - `jitter_ms`
  - `cpu_percent`
  - `memory_percent`
  - `deadline_miss`
  - `position_error_m`
  - `covariance_trace`
- Visualization script for per-run and cross-run comparison plots
- Random Forest deadline miss predictor
- Next-cycle prediction setup where `X[t]` predicts `deadline_miss[t+1]`

## Repository Structure

```text
predictive-control-loop-jitter-lab/
├── README.md
├── data/
│   └── runtime_logs/
├── models/
├── results/
│   ├── plots/
│   └── reports/
└── src/
    ├── ekf_core.py
    ├── ekf_runtime_benchmark.py
    ├── metrics_logger.py
    ├── stress_injector.py
    ├── visualize_runtime_metrics.py
    ├── train_deadline_predictor.py
    └── evaluate_predictor.py
```

## Tech Stack

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn
- psutil
- joblib

## Setup

```bash
git clone https://github.com/your-username/predictive-control-loop-jitter-lab.git
cd predictive-control-loop-jitter-lab
python -m venv venv
```

Activate the environment:

```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn psutil joblib
```

## Reproducing Benchmark Runs

Nominal EKF run:

```bash
python src/ekf_runtime_benchmark.py --duration 30 --target-period-ms 100 --stress-mode none --output data/runtime_logs/ekf_runtime_log_none.csv
```

Random delay stress:

```bash
python src/ekf_runtime_benchmark.py --duration 30 --target-period-ms 100 --stress-mode random_delay --random-delay-probability 0.35 --max-random-delay-ms 20 --output data/runtime_logs/ekf_runtime_log_random_delay.csv
```

CPU stress:

```bash
python src/ekf_runtime_benchmark.py --duration 30 --target-period-ms 100 --stress-mode cpu_stress --cpu-work-iterations 180 --output data/runtime_logs/ekf_runtime_log_cpu_stress.csv
```

Mixed stress:

```bash
python src/ekf_runtime_benchmark.py --duration 30 --target-period-ms 100 --stress-mode mixed --random-delay-probability 0.35 --max-random-delay-ms 100 --cpu-work-iterations 180 --output data/runtime_logs/ekf_runtime_log_mixed.csv
```

Generate plots from runtime logs:

```bash
python src/visualize_runtime_metrics.py data/runtime_logs/ekf_runtime_log_none.csv data/runtime_logs/ekf_runtime_log_random_delay.csv data/runtime_logs/ekf_runtime_log_cpu_stress.csv data/runtime_logs/ekf_runtime_log_mixed.csv
```

## Training the ML Model

Train the Random Forest next-cycle predictor:

```bash
python src/train_deadline_predictor.py --input-dir data/runtime_logs
```

Evaluate the trained model on an unseen runtime log:

```bash
python src/evaluate_predictor.py --input data/runtime_logs/ekf_runtime_log_mixed.csv
```

## Benchmark Results

| Scenario | Deadline Miss Rate | Average Loop Period | Average Jitter | Average Position Error |
|---|---:|---:|---:|---:|
| EKF nominal | 0.00% | 100.048 ms | 6.588 ms | 0.187 m |
| EKF random_delay | 0.33% | - | 6.722 ms | 0.166 m |
| EKF cpu_stress | 57.20% | 120.038 ms | 23.036 ms | 0.153 m |
| EKF mixed | 13.33% | 100.020 ms | 10.896 ms | 0.170 m |

## ML Results

| Model | Prediction Target | Samples | Deadline Misses | Accuracy | Precision | Recall | F1 Score |
|---|---|---:|---:|---:|---:|---:|---:|
| Random Forest | `deadline_miss[t+1]` | 1145 | 184 | 96.17% | 92.68% | 82.61% | 87.36% |

## Feature Importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `loop_execution_time_ms` | 0.37 |
| 2 | `loop_period_ms` | 0.30 |
| 3 | `jitter_ms` | 0.21 |
| 4 | `cpu_percent` | 0.09 |
| 5 | `memory_percent` | 0.03 |

## Interpretation

- The nominal EKF run holds the 100 ms target period with zero deadline misses, which validates the fixed-rate scheduler and baseline instrumentation.
- `cpu_stress` is the dominant failure mode in the current setup, pushing the average loop period to `120.038 ms` and the deadline miss rate to `57.20%`.
- `mixed` stress produces meaningful instability without fully collapsing loop timing, making it a useful scenario for training predictive models.
- EKF position error remains relatively stable across scenarios, which suggests timing degradation currently affects runtime determinism more strongly than localization accuracy.
- The Random Forest results indicate that next-cycle misses are predictable from runtime telemetry, with execution time, loop period, and jitter carrying most of the predictive signal.

## Engineering Takeaways

- Demonstrates practical real-time benchmarking for robotics software loops
- Connects systems telemetry with estimator-level quality metrics
- Shows end-to-end experimentation from workload simulation to predictive modeling
- Produces reproducible CSV, plot, model, and evaluation artifacts suitable for reporting

## Future Work

- Add a Logistic Regression baseline for simpler interpretability
- Compare against XGBoost for stronger nonlinear modeling
- Expand unseen test evaluation across separately collected benchmark runs
- Integrate the benchmark with ROS 2 nodes and executors
- Deploy on a real robot or embedded robotics compute target

## Attribution

The EKF core is adapted from the PythonRobotics project by Atsushi Sakai and contributors.
