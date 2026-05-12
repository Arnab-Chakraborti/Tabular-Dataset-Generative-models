# Tabular Dataset Generative Models

A modular PyTorch-based project for generating synthetic tabular datasets using neural generative modeling techniques.

---

# Project Setup

## 1. Clone the Repository

```bash
git clone https://github.com/Arnab-Chakraborti/Tabular-Dataset-Generative-models.git
```

Move into the project directory:

```bash
cd Tabular-Dataset-Generative-models
```

---

## 2. Create Virtual Environment

Using `uv`:

```bash
uv venv
```

Activate the environment on macOS/Linux:

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

Install all required libraries from the project configuration:

```bash
uv sync
```

---

## 4. Project Structure

```text
housing-project/
├── .venv/
├── pyproject.toml
├── uv.lock
├── README.md
├── main.py
├── data/
│   └── california_housing.csv
└── src/
    ├── __init__.py
    ├── model.py
    ├── utils.py
    └── eval_metrics.py
```

---

## 5. Running the Project

Run the main training script:

```bash
python main.py
```

---

## 6. Git Workflow

Stage changes:

```bash
git add .
```

Commit changes:

```bash
git commit -m "commit message"
```

Push to GitHub:

```bash
git push
```

---


- Automatically fetches the California Housing dataset if not present locally.
- Generated checkpoints are ignored using `.gitignore`.
- Visualization utilities are included for evaluating generated data distributions.