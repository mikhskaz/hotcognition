# Hot Cognition Lab

A cognitive modeling project simulating paired-associate learning using the pyClarion framework. This project explores how emotional factors (arousal and valence) affect memory retention and retrieval in a paired-associates task.

## Overview

This project implements a computational model of paired-associate learning using the ACT-R cognitive architecture via pyClarion. The simulation models participants learning associations between consonant-vowel-consonant (CVC) trigrams and two-digit numbers, with parameters that can be mapped to emotional dimensions.

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Required packages:
- [pyClarion](https://github.com/cmekik/pyClarion) - Cognitive architecture framework
- pandas - Data manipulation and analysis
- matplotlib - Plotting and visualization
- seaborn - Statistical data visualization
- numpy - Numerical computing

NOTE: Difficult to setup.

## Project Structure

- [simulation.py](simulation.py) - Main simulation script with model implementation
- [values.py](values.py) - Additional values and parameters
- [paired_associates.ipynb](paired_associates.ipynb) - Interactive notebook for paired-associates experiments
- [paired_associates2.ipynb](paired_associates2.ipynb) - Alternative paired-associates notebook
- [simulation_nb.ipynb](simulation_nb.ipynb) - Simulation notebook with visualization

## Model Architecture

### Agent Components

The `Participant` agent includes:
- **Input**: Receives stimuli (CVC-number pairs)
- **ChunkStore**: Declarative memory for storing associations
- **BaseLevel**: Activation-based memory decay mechanism
- **Pool**: Combines activation from multiple sources
- **Choice**: Selects responses based on activation strengths

### Key Parameters

- `scale` - Maps to valence (emotional positivity/negativity)
- `decay` - Maps to arousal (1 - arousal represents decay rate)
- `sd` - Standard deviation for choice noise
- `blw` - Base-level weight in the activation pool

## Simulation Details

### Stimuli

The model uses paired associates consisting of:
- **CVC trigrams**: Random consonant-vowel-consonant combinations (e.g., "BAC", "DEB")
- **Two-digit numbers**: Random number pairs (e.g., "23", "54")

### Procedure

1. Study phase: Present CVC-number pair
2. Test phase: Present CVC, retrieve associated number
3. Each stimulus is presented twice in randomized order
4. Inter-trial interval: 3 seconds

### Output Metrics

- **Correct recall**: Whether the retrieved number matches the target
- **Reaction time (RT)**: Computed from activation strength
- **Retention interval**: Time between study and test
- **Activation strength**: Internal measure of memory strength

## Running Simulations

### Command Line

```bash
python simulation.py
```

This will run simulations across parameter combinations and generate visualizations.

### Jupyter Notebooks

Launch Jupyter and open any of the `.ipynb` files for interactive exploration:

```bash
jupyter notebook
```

## Visualization

The project includes several plotting functions:

- `mean_delta_plot()` - Mean correct retrieval vs retention interval
- `scatter_plot()` - Decay parameter vs retention
- `heatmap()` - 2D visualization of arousal × valence effects
- `violin()` - Retrieval latency distributions by condition

## Research Questions

The simulation investigates:
- How emotional arousal affects memory decay rates
- How emotional valence influences memory activation
- The relationship between retention interval and retrieval success
- Individual differences in parameter estimation

## Future Directions

See TODOs in [simulation.py](simulation.py:525-534):
- Parameter estimation using MCMC
- Hierarchical regression analysis
- Incorporation of typing/recall time into intervals
- Interaction effects between emotional conditions
