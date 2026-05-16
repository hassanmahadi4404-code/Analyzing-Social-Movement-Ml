# Analyzing Social Movement using Neural Network in C++

A custom neural network implemented completely from scratch in C++ for analyzing social movement patterns and predicting the overall impact of a movement.

This project was developed as part of the **CSE2100 course project**.

---

# Project Idea

Social movements are influenced by many factors such as public support, inclusiveness, violence level, misinformation, social instability, and organization.

This project uses a custom-built neural network to learn the relationship between these factors and predict the overall impact of a movement.

Instead of using machine learning libraries such as TensorFlow or PyTorch, every component of the neural network was implemented manually in C++.

The project demonstrates the fundamental concepts behind machine learning:

- Data preprocessing
- Neural network architecture
- Forward propagation
- Backpropagation
- Weight updates
- Prediction
- Feature importance analysis

---

# What the Program Does

The program follows these steps:

### Step 1: Read dataset from CSV

The system reads movement data from:

```bash
Merged_Movements_final.csv
```

Each row contains:

| Feature | Meaning |
|----------|----------|
| A | Public support |
| B | Violence / Non-violence level |
| C | Inclusiveness |
| D | Organization |
| E | Property damage |
| F | Social instability |
| G | Misinformation |
| H | Overall movement impact |

---

### Step 2: Data preprocessing

The program prepares the dataset before training:

- Skips timestamp column
- Converts values into numeric form
- Normalizes input values to range:

```text
0 → 1
```

Normalization helps the neural network learn faster.

Example:

```text
Original value = 5

Normalized value:

5/5=1
```

---

### Step 3: Split training and testing data

Dataset is automatically divided:

```text
80% → Training Data

20% → Testing Data
```

Training data teaches the network.

Testing data checks how well the network learned.

---

### Step 4: Create Neural Network

The project uses this architecture:

```text
Input Layer:

7 neurons

↓

Hidden Layer:

7 neurons

↓

Output Layer:

1 neuron
```

Structure:

```text
A,B,C,D,E,F,G

↓

Hidden neurons

↓

Predicted H score
```

---

# Main Sections of Code

The code is divided into several logical sections.

---

## 1. MovementData Class

Purpose:

Stores all movement information.

Contains:

```cpp
A,B,C,D,E,F,G
```

Input factors.

and:

```cpp
H
```

Actual impact score.

Also stores:

```cpp
successScore
```

Normalized output value.

---

## 2. CSV Reader Function

Function:

```cpp
readCSVData()
```

Purpose:

Reads dataset from CSV file.

Main tasks:

- Opens file
- Reads rows
- Skips invalid data
- Normalizes values
- Stores everything into vectors

---

## 3. Neural Network Initialization

Function:

```cpp
initializeBrain()
```

Purpose:

Creates random weights and biases.

Weights are initialized randomly because neural networks begin learning from random values.

Example:

```text
Input → Hidden weights

Hidden → Output weights
```

---

## 4. Forward Propagation

Function:

```cpp
predict()
```

Purpose:

Moves information through the network.

Flow:

```text
Input Layer

↓

Hidden Layer

↓

Output Layer
```

Uses:

```cpp
Sigmoid Activation Function
```

Formula:

f(x)=1/(1+e^-x)

Purpose:

Transforms output into values between:

```text
0 and 1
```

---

## 5. Training Section

Function:

```cpp
trainBrain()
```

Purpose:

Teaches the network.

Training process:

### Forward pass

Predict output

↓

Calculate error

↓

Backward pass

↓

Update weights

↓

Repeat

The network gradually learns patterns from data.

Current parameters:

```text
Training cycles = 300

Learning rate = 0.1
```

---

## 6. Backpropagation

Purpose:

Adjust network weights using prediction error.

Process:

```text
Actual output

-

Predicted output

=

Error
```

The error travels backward through the network and updates the weights.

This helps improve future predictions.

---

## 7. Testing Section

Function:

```cpp
testBrain()
```

Purpose:

Measures model performance.

Outputs:

- Prediction
- Actual value
- Error
- Average error
- Accuracy percentage

Example:

```text
Predicted H = 7.2

Actual H = 8

Error = 0.8
```

---

## 8. Feature Importance Analysis

Function:

```cpp
analyzeFeatures()
```

Purpose:

Finds which factors affect prediction most.

Example output:

```text
Public Support: 0.65

Organization: 0.82

Misinformation: 0.40
```

Higher values indicate stronger influence.

---

## 9. Interactive Prediction System

Function:

```cpp
predictNewMovement()
```

Purpose:

Allows user to manually enter movement characteristics.

Example:

```text
Public Support:

4

Non-violence:

5

Inclusiveness:

3
```

The system predicts:

```text
Impact score

Impact category

Success probability

Recommendations
```

Example:

```text
Predicted H Score:

8.4/10

Impact Category:

Very Positive

Success Probability:

84%
```

---

# Technologies Used

- C++
- Object Oriented Programming
- STL
- File Handling
- CSV Processing
- Neural Networks
- Machine Learning Concepts

---

# Project Structure

```bash
Analyzing-Social-Movement-ML/

│
├── cse2100_project.cpp
├── Merged_Movements_final.csv
├── 2303068_custom_dataset.csv
├── README.md
└── .gitignore
```

---

# How to Run

Clone repository:

```bash
git clone https://github.com/hassanmahadi4404-code/Analyzing-Social-Movement-Ml.git
```

Compile:

```bash
g++ cse2100_project.cpp -o movement
```

Run:

```bash
./movement
```

---

# Future Improvements

Possible improvements:

- Save trained model weights
- Dynamic dataset path input
- Better visualization
- Graph generation
- Larger dataset
- GUI application
- Deep learning extension

---

# Author

Mahadi Hassan

LinkedIn:

https://www.linkedin.com/in/mahadi-hassan-b25813343/
