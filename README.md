
## 🔍 Hyperparameter Optimization using Optuna for ANN & CNN

This repository showcases how to optimize hyperparameters for Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) using [Optuna](https://optuna.org/), a powerful and flexible hyperparameter optimization framework. The project compares the performance and efficiency of Optuna with traditional methods like Grid Search and Random Search.

---

## 📂 Files Included

| File                | Description                                       |
| ------------------- | ------------------------------------------------- |
| `OPtunaonAnn.ipynb` | Hyperparameter tuning for ANN using Optuna        |
| `OPtunaonCnn.ipynb` | Hyperparameter tuning for CNN using Optuna        |


---

## 🧠 Problem Statement

In modern machine learning workflows, tuning hyperparameters is critical to achieving good model performance. Traditional methods like Grid Search and Random Search are time-consuming and often inefficient.

This project demonstrates the **use of Optuna to automatically and efficiently discover optimal hyperparameters** for:

* A basic **ANN** for image classification
* A **CNN** for more complex image datasets

---

## 🚀 Why Optuna?

| Feature                      | Grid Search                        | Random Search    | Optuna                                        |
| ---------------------------- | ---------------------------------- | ---------------- | --------------------------------------------- |
| **Exploration Method**       | Exhaustive                         | Random sampling  | Adaptive (Bayesian)                           |
| **Efficiency**               | Very low (combinatorial explosion) | Better than grid | Best (uses past trials)                       |
| **Early Stopping (Pruning)** | ❌                                  | ❌                | ✅                                             |
| **Parallelization**          | Limited                            | Limited          | ✅ Easy with RDB backend                       |
| **Ease of Use**              | Simple                             | Simple           | Requires setup, but powerful                  |
| **Search Space Flexibility** | Fixed                              | Flexible         | Highly flexible (supports conditional search) |
| **Runtime**                  | Slowest                            | Moderate         | Fastest for large spaces                      |

---

## 📈 Hyperparameters Tuned

### ANN

* Learning rate
* Number of hidden layers
* Hidden units per layer
* Dropout rate
* Optimizer

### CNN

* Number of convolutional layers
* Kernel size
* Pooling strategy
* Learning rate
* Batch size
* Dropout
* Optimizer (Adam, SGD, RMSprop)

---

## ⚙️ Setup Instructions

```bash
git clone https://github.com/yourusername/optuna-hp-tuning-ann-cnn.git
cd optuna-hp-tuning-ann-cnn

# Recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

### Requirements

```
torch
torchvision
optuna
matplotlib
numpy
pandas
scikit-learn
```

---

## 📊 Results Summary

| Model | Method | Accuracy  | Time Taken  | Best Params |
| ----- | ------ | --------- | ----------- | ----------- |
| ANN   | Grid   | 89%       | \~2 hours   | Fixed       |
| ANN   | Random | 91%       | \~1.2 hours | Varying     |
| ANN   | Optuna | **93.5%** | \~35 mins   | Adaptive    |
| CNN   | Grid   | 91.3%     | \~3.5 hours | Fixed       |
| CNN   | Random | 93.2%     | \~2.2 hours | Varying     |
| CNN   | Optuna | **95.6%** | \~55 mins   | Adaptive    |

🟢 **Optuna outperformed other methods in both accuracy and time efficiency.**

---

## 📌 How Optuna Works

Optuna uses a **define-by-run** API to dynamically construct the search space during execution. Here’s how it works:

```python
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    
    # Model building
    model = build_model(n_layers)
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr)
    
    # Train and evaluate
    accuracy = train_and_evaluate(model, optimizer)
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

---

## 🧪 Sample Visualization

```python
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

These visualizations provide insights into which hyperparameters most influence performance.

---

## 📚 Learnings

* Optuna significantly reduces the number of trials needed to reach high performance.
* Pruning unpromising trials early saves computational time.
* Dynamic search space allows more flexibility than fixed grid-based methods.

---

## 🧑‍💻 Author

* [Devesh Sharma](https://github.com/SharmajiKabetaDevesh)
* Final Year AI & Data Science Student | Passionate about ML Research

---

