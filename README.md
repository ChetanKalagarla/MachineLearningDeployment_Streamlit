
<div align="center">

# 🏡 EstateIQ
### House Price Prediction Dashboard

**A modern, interactive machine-learning dashboard for exploring property data and estimating house prices.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Linear%20Regression-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Learning%20Project-0F766E?style=for-the-badge)

**Designed with a premium navy, teal, and gold visual theme.**

</div>

---

## ✨ Project Overview

**EstateIQ** is a Streamlit-based house price prediction dashboard built using Python and machine learning. It allows users to enter property details, receive an estimated price in ₹ lakhs, review model evaluation metrics, and explore property data through interactive tables and charts.

This project demonstrates the machine-learning workflow, from data preparation and model training to prediction, evaluation, and visualization.

> **Note:** This project uses a small sample dataset of 10 records for demonstration. Predictions are educational estimates and should not be treated as actual market valuations.

## 🎯 Key Features

<table>
<tr>
<td width="50%">

### 🏠 Price Prediction

Enter property details and estimate the house price.

</td>
<td width="50%">

### 📊 Model Evaluation

Review MAE, MSE, RMSE, and R² metrics.

</td>
</tr>
<tr>
<td>

### 📈 Visual Analytics

Explore charts and patterns in property data.

</td>
<td>

### 🗂️ Dataset Explorer

View the sample dataset and download it as CSV.

</td>
</tr>
<tr>
<td>

### 🧮 Regression Insights

Explore the fitted linear regression equation.

</td>
<td>

### 🎨 Premium Interface

Use a dashboard styled with navy, teal, and gold colors.

</td>
</tr>
</table>

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data handling and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Linear Regression and evaluation metrics |

## 🧠 How It Works

```text
        🏠 Property Sample Data
                  │
                  ▼
       Prepare Features and Target
                  │
                  ▼
      Train Linear Regression Model
                  │
                  ▼
        Enter Property Details
                  │
                  ▼
      Predict Estimated House Price
                  │
                  ▼
     Explore Metrics, Charts and Data
```

### Input Features

- **Area:** Property area
- **Bedrooms:** Number of bedrooms
- **Bathrooms:** Number of bathrooms
- **Age:** Age of the property
- **Parking:** Parking value represented in the application

The model uses the input features to estimate the target house price.

### 📐 Model Evaluation Metrics

| Metric | Description |
|---|---|
| **MAE** | Average absolute difference between actual and predicted values |
| **MSE** | Average squared prediction error |
| **RMSE** | Prediction error expressed in the target's units |
| **R² Score** | Indicates how well the model explains variation in the target |

## 🚀 Run the Project Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/ChetanKalagarla/MachineLearningDeployment_Streamlit.git
cd MachineLearningDeployment_Streamlit
```

### Step 2: Create a Virtual Environment

For Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r Requirements.txt
```

### Step 4: Launch the Application

```bash
streamlit run App.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## 📁 Project Structure

```text
MachineLearningDeployment_Streamlit/
│
├── App.py
│   └── Streamlit dashboard and machine-learning workflow
│
├── Requirements.txt
│   └── Required Python packages
│
└── README.md
    └── Project documentation
```

## 📌 Project Scope and Limitations

- The application uses 10 sample records.
- A small dataset is not sufficient for reliable real-world predictions.
- Results depend on the available data and input values.
- The project is intended for learning and demonstration.
- Model evaluation results should be interpreted cautiously because the dataset is very small.

## 🔮 Future Improvements

- Integrate a larger, real-world housing dataset.
- Compare Linear Regression with other regression algorithms.
- Add train-test splitting and cross-validation.
- Improve input validation and error handling.
- Add location-based property features.
- Deploy the application for public access.

## 🔗 GitHub Repository

Explore the source code:

**[MachineLearningDeployment_Streamlit](https://github.com/ChetanKalagarla/MachineLearningDeployment_Streamlit)**

## 👨‍💻 Author

**Chetan Kalagarla**

Built as a practical project to explore machine learning, data analytics, and Streamlit application development.

---

<div align="center">

### 🏡 EstateIQ

**Explore the Data. Understand the Model. Estimate with Insight.**

*Made with Python, Streamlit, and Machine Learning.*

</div>
