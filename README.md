#  Social Movement Analysis using a Custom Neural Network in C++

[![C++](https://img.shields.io/badge/Language-C%2B%2B%2017-blue.svg)](https://isocpp.org/)
[![Course Project](https://img.shields.io/badge/Course%20Project-CSE2100-red.svg)]()


A lightweight, high-performance neural network implemented **completely from scratch** in C++ to analyze social movement patterns and predict their overall impact. Developed as part of the **CSE2100** course project.

---

## Project Overview

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

1. Data Pipeline & FeaturesThe application processes inputs from Merged_Movements_final.csv. Each records represents a historical social movement containing the following quantified attributes:Feature IDAttribute NameDescriptionAPublic SupportLevel of general population backingBNon-Violence LevelDegree of peaceful protest vs. violent escalationsCInclusivenessDiversity of demographic participationDOrganizationCoordination quality and leadership structureEProperty DamageExtent of physical/infrastructure destructionFSocial InstabilityPre-existing friction/chaos in the regionGMisinformationProliferation of fake news/counter-propagandaH (Target)Overall ImpactThe ground-truth target score to predict2. Mathematics & Network TopologyThe system instantiates a 7 ➔ 7 ➔ 1 Dense Feedforward Network architecture.Activation Function: Sigmoid is utilized to map values smoothly between the $0$ and $1$ boundaries.$$f(x) = \frac{1}{1 + e^{-x}}$$Data Normalization: Simple, efficient linear transformation scaled down to prevent neuron saturation and gradient vanishing issues.🛠️ Codebase ArchitectureThe project code is cleanly encapsulated into modular sections:💾 MovementData ClassAn Object-Oriented representation of a movement profile. It handles data storage, internal state transitions, and manages its own normalized feature maps.📊 Mathematical & Training SubsystemsreadCSVData(): Secure file parsing stream that skips metadata headers, handles null-safety checks, maps values to vectors, and normalizes them dynamically.initializeBrain(): Uses a randomized distribution to seed network weights and biases, avoiding the symmetrical weight trap.predict() (Forward Propagation): Calculates the dot product of weights and inputs, applies biases, and pushes activations sequentially through the hidden layer to the output node.trainBrain() (Backpropagation): Evaluates network error utilizing Mean Squared Error logic, calculates local gradients via the chain rule derivative of the Sigmoid function, and runs gradient descent optimizations across 300 cycles at a learning rate ($\alpha$) of 0.1.testBrain(): Runs unseen testing data subsets through the system, outputting absolute metrics, Mean Absolute Error (MAE), and final accuracy thresholds.📈 Analytics & Live Interactive LayeranalyzeFeatures(): Evaluates the finalized weights matrices to isolate which input factors exert the strongest mathematical pull on predictions.predictNewMovement(): A dynamic, CLI-driven environment prompting users for custom operational criteria to return real-time predictions, classification tags, confidence ratings, and proactive strategic recommendations.📂 Repository StructureBashAnalyzing-Social-Movement-ML/
│
├── cse2100_project.cpp          # Core C++ source implementation file
├── Merged_Movements_final.csv    # Principal evaluation dataset
├── 2303068_custom_dataset.csv   # Supplementary/Validation dataset
├── README.md                    # Project documentation
└── .gitignore                   # Build artifact tracking configuration
🚀 Getting StartedPrerequisitesEnsure you have a modern C++ compiler installed (g++ supporting C++11 or higher).Compilation & ExecutionClone the repository to your local system:Bashgit clone [https://github.com/hassanmahadi4404-code/Analyzing-Social-Movement-Ml.git](https://github.com/hassanmahadi4404-code/Analyzing-Social-Movement-Ml.git)
cd Analyzing-Social-Movement-Ml


2. Compile using a standard optimized release configuration:
   ```bash
   g++ -O3 cse2100_project.cpp -o movement_analyzer
   
Launch the binary executable:Bash./movement_analyzer


---

## 🔮 Roadmap & Future Scope
- [ ] Implement serialization to save and load trained weight configurations (`.bin` / `.txt`).
- [ ] Add support for dynamically configurable hidden layer counts and sizes.
- [ ] Expand parameter optimization algorithms (e.g., Adam or Momentum-based gradient descent).
- [ ] Design a lightweight cross-platform Desktop GUI interface.

---

## 👤 Author

**Mahadi Hassan**
*   **Institution:** Rajshahi University of Engineering and Technology (RUET)
*   **Department:** Computer Science and Engineering (CSE)
*   **LinkedIn:** [@mahadi-hassan](https://www.linkedin.com/in/mahadi-hassan-b25813
