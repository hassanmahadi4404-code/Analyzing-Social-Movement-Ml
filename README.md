# 🧠 Social Movement Analysis using a Custom Neural Network in C++

[![C++](https://img.shields.io/badge/Language-C%2B%2B%2017-blue.svg)](https://isocpp.org/)
[![Course Project](https://img.shields.io/badge/Course%20Project-CSE2100-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A lightweight, high-performance neural network implemented **completely from scratch** in C++ to analyze social movement patterns and predict their overall impact. Developed as part of the **CSE2100** course project.

---

## 📌 Project Overview

Social movements are complex phenomena driven by multi-dimensional factors such as public support, organizational structure, media landscape, and socioeconomic stability. This project aims to model and predict the overall impact score of a social movement by training a custom multi-layer neural network on historical data.

### Why Build from Scratch?
Instead of relying on heavy framework abstractions like TensorFlow or PyTorch, every underlying mechanism—from matrix-like weights manipulation to calculus-based backpropagation—was written in pure C++. This provides a granular understanding of:
*   Memory management and optimization in mathematical models.
*   The exact mechanics of gradient descent and weight updates.
*   Data pipeline structures inside production-level C++.

---

## ⚙️ How the Engine Works

The core workflow of the system is engineered into 4 distinct phases:

```text
[ CSV Dataset ] ──> [ Preprocessing & Min-Max Normalization ] ──> [ 80/20 Train/Test Split ]
                                                                             │
[ Interactive CLI Predictions ] <── [ Feature Importance Analysis ] <── [ Network Training ]
