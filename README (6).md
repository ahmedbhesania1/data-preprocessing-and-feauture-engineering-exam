# Customer Purchase Propensity — Data Cleaning & Feature Engineering Pipeline

**Practical Exam — Data Preprocessing and Feature Engineering (Red & White Skill Education)**

A full preprocessing pipeline built as a Junior Data Analyst at an e-commerce company: raw customer, transaction and product data from four different sources is merged, explored, cleaned, encoded, scaled and engineered into a single ML-ready dataset. The pipeline is built to feed a future binary classification model that predicts whether a customer will make a purchase.

---

## Table of Contents

1. [Objective](#objective)
2. [Dataset](#dataset)
3. [Tech Stack](#tech-stack)
4. [Project Workflow](#project-workflow)
5. [Assumptions Made](#assumptions-made)
6. [Key Findings](#key-findings)
7. [Deliverables](#deliverables)
8. [How to Run](#how-to-run)
9. [Project Structure](#project-structure)
10. [Author](#author)

---

## Objective

Clean, preprocess and engineer meaningful features from multi-source customer and purchase data so it is ready for a machine learning model, without training the model itself. The brief frames this as a binary classification problem: predict whether a customer will make a purchase or not.

## Dataset

Four sources were loaded and merged into a single working dataset.

| Source | File / Endpoint | Rows | Contains |
|---|---|---|---|
| CSV | `customers.csv` | 100 | customer_id, name, age, gender, city, income |
| JSON | `transactions.json` | 100 | transaction_id, customer_id, product_id, amount, payment_mode, date |
| SQL | `products.sql` (MySQL table `products`) | 100 | product_id, product_name, category, price, stock |
| API | `https://dummyjson.com/users` | 100 | additional user details (reference only) |

Customers and transactions were merged on `customer_id`, then merged with products on `product_id`, giving a transaction-level dataset of **100 rows and 15 columns**. The merged data had **no missing values**, so Step 4 (missing data handling) was not needed for this run.

## Tech Stack

| Tool | Use |
|---|---|
| Python 3 | Programming language |
| Pandas | Loading, merging, cleaning and transforming data |
| NumPy | Numerical operations and transformations |
| Matplotlib / Seaborn | Histograms, scatter plots, correlation heatmap |
| YData Profiling | Automated EDA report |
| scikit-learn | `LabelEncoder`, `OrdinalEncoder`, scalers, `FunctionTransformer`, `PowerTransformer`, `ColumnTransformer` |
| MySQL Connector | Reading the products table from SQL |
| Requests | Fetching user data from the API |
| Jupyter Notebook | Development environment |

## Project Workflow

All code is in [`main_1.ipynb`](main_1.ipynb), covering all 10 steps of the exam brief.

### Step 1: Project Planning & Problem Framing
- **Data analysis** is the process of inspecting, cleaning and modelling data to find useful patterns and support decisions.
- **Data science project steps:** define the problem, collect data, understand and clean it, explore it (EDA), engineer features, model, evaluate, deploy, and communicate results.
- **ML framing:** supervised binary classification — predict whether a customer will make a purchase (`purchased` = 1 or 0) using demographic and transaction features.

### Step 2: Data Import and Understanding
- Loaded the CSV, JSON, SQL and API sources separately with `pd.read_csv`, `pd.read_json`, `pd.read_sql_query` (via `mysql.connector`) and `requests.get`.
- Merged customers and transactions on `customer_id`, then merged the result with products on `product_id`.
- Explored the combined dataset with `.info()`, `.describe()` and `.head()` — 100 rows, 15 columns, no missing cells.

### Step 3: Exploratory Data Analysis
- **Univariate:** histograms for `age`, `income`, `amount`, `price` and `stock`. `amount`, `price` and `stock` came out right-skewed. A YData Profiling report (`report.html`) was generated for automated insights.
- **Bivariate:** scatter plot of `income` vs `amount` to look for a spending relationship.
- **Multivariate:** correlation heatmap across all numeric columns.

### Step 4: Handling Missing Data
Not applicable — the merged dataset had zero missing values in every column, confirmed with `.isnull().sum()`.

### Step 5: Outlier Detection & Handling
- **Z-score method** on `amount` — rows with `|z-score| ≥ 3` removed.
- **IQR method** on `price` — rows outside `Q1 − 1.5×IQR` to `Q3 + 1.5×IQR` removed.

### Step 6: Handling Mixed & Date/Time Variables
- `date` converted to `datetime`; `days_since_last_purchase` derived as the gap between today and the transaction date.
- `transaction_id` and `product_id` (mixed alphanumeric, e.g. `T001`, `P001`) split into a letter part and a number part.
- The letter columns (`t_id_letter`, `p_id_letter`) turned out to be constant (`T` and `P` for every row) and were dropped as they carry no information. The original `transaction_id` and `product_id` columns were dropped after splitting.

### Step 7: Encoding Categorical Data
- **Label encoding:** `city` and `gender`.
- **One-hot encoding:** `payment_mode` and `category`.
- **Ordinal encoding:** the dataset had no natural ordinal column (like education or satisfaction level), so `income` was binned into ordered groups — `low`, `medium`, `high`, `very high` — and ordinal-encoded in that order.
- **Encoding numerical features:** the income-binning above doubles as the "binning income groups" requirement.

### Step 8: Feature Scaling
Each scaler was applied to a single numeric column to show its individual effect, then combined using `ColumnTransformer` to apply several at once:

| Scaler | Applied to |
|---|---|
| StandardScaler | `age` |
| MinMaxScaler | `income` |
| MaxAbsScaler | `amount` |
| RobustScaler | `price` |
| Normalizer | `stock` |

### Step 9: Feature Construction & Transformation
- `total_purchases` — number of transactions per customer.
- `days_since_signup` — days since each customer's first transaction (used as a signup-date proxy, see [Assumptions](#assumptions-made)).
- `purchase_per_day` = `total_purchases / days_since_signup`.
- **FunctionTransformer:** log (`amount`), square root (`price`), reciprocal (`stock`).
- **PowerTransformer:** Yeo-Johnson on `amount`, Box-Cox on `price` — both were flagged as right-skewed in Step 3.
- **Quantile binning:** `income` split into 4 equal-sized bins (`q1`–`q4`).
- **Binarization:** `frequent_buyer` = 1 if a customer's `total_purchases` is above the median, else 0.

### Step 10: Final Output
The cleaned, encoded, scaled and feature-engineered dataset is exported to `processed_customer_data.csv`.

## Assumptions Made

The dataset provided doesn't contain a few columns the original brief assumes, so the following stand-ins were used:

- **No `purchased` target column:** every row is already a completed transaction, so there's no built-in churn-style 0/1 label in this data. Bivariate analysis was done against `income`/`amount` instead.
- **No `signup_date`:** each customer's earliest transaction `date` was used as a proxy signup date to calculate `days_since_signup`.
- **No education/satisfaction column:** `income`, binned into `low`/`medium`/`high`/`very high`, was used to demonstrate ordinal encoding instead.

## Key Findings

- The merged dataset was clean going in — no missing values and no exact duplicates.
- `amount`, `price` and `stock` are right-skewed; log and Yeo-Johnson/Box-Cox transforms reduce the skew.
- `t_id_letter` and `p_id_letter` carried zero information (constant values) and were dropped rather than encoded.
- Outlier handling with z-score and IQR removed a small number of extreme rows from `amount` and `price` without materially shifting the overall distribution.
- Of the scalers tried, `RobustScaler` is the most appropriate default for this data given the right-skewed, outlier-prone numeric columns; `StandardScaler`/`MinMaxScaler` are better suited to the more symmetric columns like `age`.

## Deliverables

| Requirement | Status |
|---|---|
| Python Notebook (`main_1.ipynb`) | Ready |
| Final CSV output (`processed_customer_data.csv`) | Ready |
| Summary report (theory + observations, max 1 page) | Ready — see [Assumptions](#assumptions-made) and [Key Findings](#key-findings) |
| GitHub repository with README.md | This file |

## How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```
2. Install dependencies
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn requests mysql-connector-python ydata-profiling jupyter
   ```
3. Update the file paths in `main_1.ipynb` (currently absolute Windows paths) so they point to `customers.csv` and `transactions.json` in your copy of the repository.
4. *(Optional, for the SQL step)* Create a MySQL database named `dp_exam` with a `products` table, and update the host, user and password in the connection cell.
5. Launch Jupyter and run all cells from top to bottom
   ```bash
   jupyter notebook main_1.ipynb
   ```
6. The final processed dataset is saved as `processed_customer_data.csv` in the same folder.

## Project Structure

```
.
├── README.md
├── main_1.ipynb                     # full preprocessing & feature engineering notebook
├── customers.csv                    # raw customer data
├── transactions.json                # raw transaction data
├── products.sql                     # raw product data (SQL)
├── report.html                      # YData Profiling report
└── processed_customer_data.csv      # final cleaned, encoded, scaled dataset
```

## Author

**Ahmed**
