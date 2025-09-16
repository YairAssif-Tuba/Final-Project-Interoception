# prj_interoception_modeling
Explores how cardiac properties, especially HRV, influence neural time perception mechanisms. Based on Bi &amp; Zhou's computational framework (PNAS, 2020), we investigate how cardiac signals shape temporal processing in RNNs.

## Neural Network Models for Temporal Processing

This repository contains an implementation of the neural network architecture described in:

Bi, Z., & Zhou, C. (2020). "Understanding the computational principles of time using neural network models." arXiv:1910.05546

## Overview

This codebase provides a PyTorch implementation of recurrent neural networks specialized for temporal processing tasks, particularly interval production and interval comparison. The project explores computational mechanisms underlying temporal processing in neural circuits.

## Project Structure

- **my_rnn/**: Core implementation of the neural network architecture
  - `network.py`: RNN model with specialized temporal processing features
  - `task.py`: Task definitions for interval timing experiments
  - `dataset.py`: Data handling utilities
  - `train_stepper.py`: Training procedures
  - `test_modules.py`: Testing and evaluation utilities

- **IntervalTiming/**: Original reference implementation by Bi and Zhou
  - Contains scripts to reproduce figures from the paper
  - Includes different model variants for specific experiments
  - Provides pretrained models and analysis scripts

## Features

- **Specialized RNN Architecture**: Implements noise-injected dynamics and leaky integration for robust temporal processing
- **Temporal Tasks**: Support for interval production, interval comparison, and time bisection tasks
- **Configurable Training**: Flexible hyperparameter configuration for neural network training
- **Analysis Tools**: Utilities for analyzing trained networks and visualizing results

## Dependencies

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib (for visualization)
- scikit-learn (for analysis)
- numba (for optimization)
- psignifit (for time bisection task psychometric function fitting)

## Getting Started

EDIT LATER 

## Reproducing Paper Results

The original implementation in the IntervalTiming directory can be used to reproduce figures from the Bi & Zhou paper:

```
python IntervalTiming/fig2.py
python IntervalTiming/fig3.py
python IntervalTiming/fig4.py
python IntervalTiming/fig5.py
python IntervalTiming/fig6.py
```

## Citation

If you use this code in your research, please cite:

```
@article{bi2020understanding,
  title={Understanding the computational principles of time using neural network models},
  author={Bi, Zedong and Zhou, Changsong},
  journal={Proceedings of the National Academy of Sciences},
  year={2020}
}
```
