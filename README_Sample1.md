Here is the revised `README.md` tailored for a **single Jupyter Notebook** workflow:

---

```markdown
# Basketball EPV Tactical Decision Engine

## Project Overview

This project implements a **Expected Possession Value (EPV)** based basketball tactical decision engine within a single Jupyter Notebook. The system evaluates the expected scoring value of different decisions (shoot, pass, drive) during each offensive possession, providing quantitative evidence for tactical analysis. The study includes **1000+ game experiments** to validate the robustness and tactical realism of the decision model under various game contexts.

The notebook contains two core models:
- **Tree-based Models** (RandomForest + XGBoost): Traditional EPV estimation based on static features
- **LSTM Sequence Model**: Temporal EPV estimation based on offensive possession event sequences

Additionally, the notebook provides comprehensive visualisation tools, including player decision value evaluation, tactical spatial heatmaps, and an interactive Dash dashboard.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Dependencies](#dependencies)
- [Installation & Setup](#installation--setup)
- [Notebook Structure](#notebook-structure)
- [Dataset Description](#dataset-description)
- [Running Guide](#running-guide)
- [Outputs & Results](#outputs--results)
- [Citation & Acknowledgements](#citation--acknowledgements)

---

## System Requirements

- **Python**: 3.9 or higher
- **Operating System**: Windows 10/11, macOS, or Linux
- **Memory**: 8GB+ recommended (LSTM training and large-scale data processing)
- **GPU**: Optional; LSTM supports CUDA acceleration
- **Jupyter Environment**: Jupyter Notebook or JupyterLab

---

## Dependencies

Install the following packages before running the notebook:

```text
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
torch>=1.10.0
joblib>=1.0.0
matplotlib>=3.4.0
plotly>=5.0.0
dash>=2.0.0
tqdm>=4.62.0
xgboost>=1.5.0
```

> **Note**: `xgboost` is optional. If not installed, only the RandomForest model will be used.

---

## Installation & Setup

### 1. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

Save the dependency list above as `requirements.txt`, then run:

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install pandas numpy scikit-learn torch joblib matplotlib plotly dash tqdm xgboost
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

### 4. Prepare the Data

Place the NBA play-by-play data file `cdnnba_2022.csv` in the same directory as the notebook. The file should contain the following fields:

`gameId`, `actionNumber`, `orderNumber`, `clock`, `period`, `actionType`, `subType`, `personId`, `playerName`, `teamTricode`, `description`, `scoreHome`, `scoreAway`, `x`, `y`, `shotDistance`, `assistPersonId`, `blockPersonId`

---

## Notebook Structure

The entire project is contained in **one Jupyter Notebook** (e.g., `nba_epv_engine.ipynb`), organised into the following logical sections:

| Section | Description |
|---|---|
| **Cell 1: Imports & Configuration** | Global imports, constants (`RAW_COLUMNS`, `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`), and random seed setup |
| **Cell 2: Preprocessing Functions** | Data cleaning, possession splitting, court coordinate normalisation, time/score feature engineering, defensive pressure calculation |
| **Cell 3: Tree Model Functions** | `build_modeling_frame()`, model training with RandomForest/XGBoost, GroupKFold cross-validation, feature importance extraction |
| **Cell 4: LSTM Model Functions** | Sequence dataset construction, `LSTMEPVModel` class definition, masked MSE loss, training loop with early stopping |
| **Cell 5: Action Value Report** | `build_action_value_report()`: counterfactual Q-value estimation, decision value added (DVA), regret calculation |
| **Cell 6: Baseline Evaluation** | Shot Chart spatial baseline evaluation using the same train/test split as the main model |
| **Cell 7: Visualisation** | Tactical-zone EPV spatial decision map (Plotly), shot regret maps, decision shift arrows |
| **Cell 8: Dashboard** | Dash interactive dashboard setup for player/team filtering and EPV heatmaps |
| **Cell 9: Paper Pack Generator** | Auto-generation of thesis figures, tables, and player report cards |
| **Cell 10: Main Execution** | `run_tree_pipeline()`, `run_lstm_pipeline()`, `generate_paper_pack()`, `launch_dashboard()` |

---

## Dataset Description

### Data Source

This project uses **public NBA play-by-play data**, file name `cdnnba_2022.csv`.

- **Data Type**: Public game event-level data
- **Content**: 2022 NBA regular season play-by-play records, including event descriptions, player information, coordinate positions, score changes, etc.
- **Acquisition**: Available via public NBA data APIs or third-party sports data platforms
  - NBA Stats API: https://stats.nba.com/
  - https://github.com/shufinskiy/nba_data/blob/main/datasets/cdnnba_2022.tar.xz
- **Format Requirements**: CSV format; fields must match the `RAW_COLUMNS` configuration


---

## Running Guide

Execute the notebook **sequentially from top to bottom**. Do not skip cells, as later cells depend on variables defined in earlier ones.

### Step-by-Step Execution

```python
# Cell 1-2: Run imports and preprocessing functions

# Cell 10 (or designated execution cell): Run the full pipeline
tree_dir, processed_df = run_tree_pipeline(
    raw_csv_path="cdnnba_2022.csv",
    n_games=1000,        # Set to None to process all games
    cv_splits=3,
)

# LSTM pipeline
lstm_dir = run_lstm_pipeline(processed_df, epochs=8)

# Generate thesis material pack
generate_paper_pack(tree_dir, lstm_dir, output_dir=Path("paper_pack"))

# Launch dashboard (optional)
launch_dashboard(tree_dir, port=8054, debug=True)
```

### Key Execution Notes

- **Do not reset the kernel** between the tree pipeline and LSTM pipeline; `processed_df` is passed directly to the LSTM function
- If you encounter CUDA memory issues during LSTM training, set `device_str="cpu"` in `run_lstm_pipeline()`
- For quick testing, set `n_games=50` in `run_tree_pipeline()`

---

## Outputs & Results

After executing the notebook, the following directories will be created automatically:

| Output | Location | Description |
|---|---|---|
| `model_metrics.csv` | `artifacts/enhanced_*/` | Model performance metrics (RMSE, MAE, R²) |
| `feature_importance.csv` | `artifacts/enhanced_*/` | Feature importance rankings |
| `scored_events.csv` | `artifacts/enhanced_*/` | All events with EPV predictions |
| `shot_action_value_report.csv` | `artifacts/enhanced_*/` | Shot-level decision value report |
| `shot_player_decision_summary.csv` | `artifacts/enhanced_*/` | Player-level decision summary |
| `lstm_metrics.csv` | `artifacts/lstm_*/` | LSTM validation/test metrics |
| `lstm_history.csv` | `artifacts/lstm_*/` | Training epoch history |
| `paper_pack/` | `./paper_pack/` | Thesis-ready figures, tables, and player report cards |

### Visualisations Generated

- **Model Comparison Chart**: Tree vs. LSTM RMSE comparison
- **Feature Importance Plot**: Top 20 most important features
- **Tactical Zone Map**: Interactive Plotly chart with Paint (2pt) and Corner/Arc (3pt) recommendations
- **Player Report Cards**: Regret maps and decision shift arrows for case-study players (e.g., Curry, Harden, Tatum)

---

## Citation & Acknowledgements

The code for this project was written independently by the author. Generative AI tools were used to assist with code structure design and documentation drafting.

