# Spike-HighGamma Analysis

A comprehensive analysis pipeline for investigating the relationship between neural spike activity and high-gamma band activity (HGA) in non-human primates during motor control tasks.

## Overview

This repository contains tools and analysis notebooks for studying neural population activity recorded from multi-electrode arrays implanted in motor cortex. The project focuses on analyzing the coupling between spike trains and high-gamma band activity across electrode arrays during behavioral performance.

## Project Structure

```
spike-highgamma/
├── behavior_analysis.ipynb           # Behavioral performance and cursor trajectory analysis
├── neural_signal_analysis.ipynb      # Neural signal correlation analysis
├── STA_analysis.ipynb                # Spike-triggered average analysis
├── subspace_analysis.ipynb           # Dimensionality reduction and subspace analysis
├── utility_functions.py              # Core utility functions and data loading
├── environment.yaml                  # Conda environment specification
└── data/                            # Data directory (excluded from version control)
    ├── trials/                      # Full trial data
    ├── last4s/                      # Last 4 seconds of trials
    ├── STA/                         # Spike-triggered average data
    ├── metadata/                    # Electrode mapping and metadata
    └── shunted_electrodes.mat       # Information about shunted electrodes
```

## Features

### Analysis Notebooks

1. **Behavior Analysis** ([behavior_analysis.ipynb](behavior_analysis.ipynb))
   - Cursor trajectory visualization
   - Success rate analysis
   - Trial-by-trial behavioral performance
   - Day-by-day behavioral trends

2. **Neural Signal Analysis** ([neural_signal_analysis.ipynb](neural_signal_analysis.ipynb))
   - Correlation between spike activity and high-gamma band activity
   - Spatial distribution analysis across electrode arrays
   - Temporal dynamics during task performance
   - Neural-behavioral correlation analysis

3. **Spike-Triggered Average Analysis** ([STA_analysis.ipynb](STA_analysis.ipynb))
   - Spike-triggered average (STA) computation
   - Factor analysis of neural population activity
   - Clustering analysis of neural patterns
   - Spatial mapping of STA features

4. **Subspace Analysis** ([subspace_analysis.ipynb](subspace_analysis.ipynb))
   - Principal Component Analysis (PCA)
   - Factor Analysis
   - Contrastive PCA (cPCA) for identifying task-relevant neural dimensions
   - Linear regression for neural decoding

### Utility Functions

The `utility_functions.py` module provides essential functions for:

- **Data Loading**: Load trial data, STA data, and metadata from `.mat` files
- **Electrode Mapping**: Handle electrode grid layouts and coordinate transformations
- **Spatial Analysis**: Calculate distance-based weights and visualize data on electrode grids
- **Statistical Utilities**: Significance testing and data preprocessing
- **Dimensionality Reduction**: Custom implementations of cPCA and covariance analysis

## Installation

### Prerequisites

- Python 3.8
- Conda (recommended for environment management)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/caraido/spike-highgamma.git
cd spike-highgamma
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yaml
conda activate spike-highgamma
```

### Dependencies

- **numpy**: Numerical computations
- **scipy**: Scientific computing and signal processing
- **scikit-learn**: Machine learning and dimensionality reduction
- **pandas**: Data manipulation and analysis
- **mat73**: Loading MATLAB v7.3 files
- **matplotlib**: Data visualization
- **contrastive**: Contrastive PCA implementation

## Data Structure

The analysis expects data organized by monkey subjects (identified by letters: C=Chewie, J=Jaco, M=Mini) and electrode numbers. Each trial contains:

- **spike_rate**: Neural firing rates
- **HGA**: High-gamma band activity (filtered and processed)
- **position**: Cursor position trajectories
- **velocity**: Cursor velocity
- **trial_types**: Task condition labels
- **target_types**: Target location information
- **file_types**: Session type (learned/adaptive)

## Usage

### Basic Analysis Workflow

1. **Load Data**:
```python
from utility_functions import load_ONF_data, get_grid, load_shunted_electrodes

# Specify parameters
data_type = 'trials'  # Options: 'trials', 'last4s', 'STA'
monkey = 'C'          # Options: 'C' (Chewie), 'J' (Jaco), 'M' (Mini)
CE = 63               # Channel electrode number

# Load data
raw_data = load_ONF_data(data_type, monkey, CE)
grid = get_grid(monkey)
shunted_electrodes = load_shunted_electrodes(monkey, spike_channel)
```

2. **Run Analysis**:
   - Open any of the Jupyter notebooks
   - Update the `monkey` and `CE` parameters as needed
   - Execute cells sequentially to reproduce analyses

3. **Visualize Results**:
```python
from utility_functions import plot_on_grid

# Visualize data on electrode grid
plot_on_grid(ax, leftover_electrodes, grid, data, loc=(loc_x, loc_y))
```

## Key Analysis Methods

### Spike-Triggered Average (STA)
Computes the average high-gamma activity surrounding each spike event to identify temporal patterns of neural population activity.

### Contrastive PCA (cPCA)
Identifies neural subspaces that distinguish between different task conditions (e.g., foreground vs. background activity, different movement directions).

### Spatial Decay Analysis
Models the spatial spread of neural activity across the electrode array using distance-based decay functions:
```
weight = 1 / d^power
```
where `d` is the Euclidean distance between electrodes.

## Data Availability

Due to file size constraints and data sharing agreements, the `data/` directory is excluded from version control. Please contact the repository maintainer for data access.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Citation

If you use this code in your research, please cite the associated publication (to be added).

## License

[License information to be added]

## Contact

For questions or data access requests, please open an issue on GitHub or contact the repository maintainer.
