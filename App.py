
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="EstateIQ | House Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 2. PREMIUM DESIGN
# ==========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #F7F9FC, #EAF1F7);
    color: #172033;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101A2C, #19334D);
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

[data-testid="stSidebar"] input {
    background: #FFFFFF !important;
    color: #172033 !important;
}

h1, h2, h3 {
    color: #101A2C !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 800 !important;
}

.hero {
    background: linear-gradient(115deg, #101A2C, #19334D, #0F766E);
    padding: 32px;
    border-radius: 22px;
    margin-bottom: 20px;
    box-shadow: 0 12px 30px rgba(16,26,44,0.16);
}

.hero .kicker {
    color: #E8B45B;
    letter-spacing: 3px;
    font-size: 12px;
    font-weight: 800;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 2.5rem;
    margin: 10px 0;
}

.hero p {
    color: #D7E3EF;
    font-size: 1rem;
    margin-bottom: 0;
}

.section-label {
    color: #0F8D83;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}

.price-card {
    background: linear-gradient(130deg, #0F766E, #0F9D91, #43B9A7);
    color: white;
    padding: 28px;
    border-radius: 20px;
    box-shadow: 0 10px 28px rgba(15,118,110,0.2);
}

.price-card .label {
    color: #D9FFFA;
    font-size: 14px;
    font-weight: 600;
}

.price-card .value {
    color: #FFFFFF;
    font-family: 'Manrope', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    margin: 12px 0;
}

.price-card .foot {
    color: #E0FFFA;
    font-size: 13px;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DDE5EF;
    border-radius: 15px;
    padding: 16px;
    box-shadow: 0 4px 14px rgba(16,26,44,0.05);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #667085 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #101A2C !important;
    font-weight: 800 !important;
}

div.stButton > button,
div[data-testid="stDownloadButton"] button {
    background: #101A2C !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: 700 !important;
}

div.stButton > button:hover,
div[data-testid="stDownloadButton"] button:hover {
    background: #0F9D91 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #E9EEF5;
    padding: 7px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    color: #344054;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    color: #0F766E !important;
    background: white !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #DDE5EF;
    border-radius: 12px;
}

.info-strip {
    background: white;
    border-left: 4px solid #E8B45B;
    border-radius: 10px;
    padding: 14px;
    margin: 15px 0;
    color: #475467;
}

.footer {
    text-align: center;
    color: #667085;
    padding: 15px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 3. SAMPLE DATASET
# ==========================================================
@st.cache_data
def load_data():
    return pd.DataFrame({
        "Area_sqft": [
            800, 1000, 1200, 1500, 1800,
            2000, 2200, 2500, 2800, 3000
        ],
        "Bedrooms": [2, 2, 2, 3, 3, 3, 4, 4, 4, 5],
        "Bathrooms": [1, 2, 2, 2, 3, 3, 3, 3, 4, 4],
        "Age_years": [20, 15, 12, 10, 8, 7, 5, 4, 3, 2],
        "Parking": [0, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "Price_lakhs": [
            30, 40, 48, 60, 72,
            82, 95, 110, 128, 145
        ]
    })


df = load_data()

FEATURES = [
    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "Age_years",
    "Parking"
]

X = df[FEATURES]
y = df["Price_lakhs"]

# ==========================================================
# 4. TRAIN AND EVALUATE MODEL
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# ==========================================================
# 5. HEADER
# ==========================================================
st.markdown("""
<div class="hero">
    <div class="kicker">ESTATEIQ | PROPERTY INTELLIGENCE</div>
    <h1>🏡 House Price Prediction</h1>
    <p>Explore property data, evaluate machine learning
    performance, and estimate house prices using
    Multiple Linear Regression.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-strip">
<b>Demo dataset:</b> This dashboard uses 10 illustrative
properties. Predictions are for educational purposes and
should not be used as professional property valuations.
</div>
""", unsafe_allow_html=True)

# ==========================================================
# 6. SIDEBAR INPUTS
# ==========================================================
st.sidebar.title("🏡 Property Studio")
st.sidebar.write("Customize the property details below.")
st.sidebar.divider()

area = st.sidebar.number_input(
    "Built-up area (sq. ft.)",
    min_value=300, max_value=10000,
    value=2100, step=100
)

bedrooms = st.sidebar.number_input(
    "Bedrooms", min_value=1, max_value=10,
    value=3, step=1
)

bathrooms = st.sidebar.number_input(
    "Bathrooms", min_value=1, max_value=10,
    value=3, step=1
)

age = st.sidebar.number_input(
    "Property age (years)",
    min_value=0, max_value=100,
    value=5, step=1
)

parking = st.sidebar.number_input(
    "Parking spaces",
    min_value=0, max_value=10,
    value=2, step=1
)

input_data = pd.DataFrame([{
    "Area_sqft": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Age_years": age,
    "Parking": parking
}])

predicted_price = float(
    model.predict(input_data[FEATURES])[0]
)

st.sidebar.divider()
st.sidebar.subheader("Selected Property")
st.sidebar.write(f"**Area:** {area:,} sq. ft.")
st.sidebar.write(f"**Bedrooms:** {bedrooms}")
st.sidebar.write(f"**Bathrooms:** {bathrooms}")
st.sidebar.write(f"**Age:** {age} years")
st.sidebar.write(f"**Parking:** {parking} spaces")

# ==========================================================
# 7. PRICE PREDICTION
# ==========================================================
st.markdown(
    '<div class="section-label">01 / PROPERTY ESTIMATE</div>',
    unsafe_allow_html=True
)
st.markdown("## Your Estimated Property Value")

left, right = st.columns([1.3, 1], gap="large")

with left:
    st.markdown(f"""
    <div class="price-card">
        <div class="label">ESTIMATED HOUSE PRICE</div>
        <div class="value">₹ {predicted_price:,.2f} Lakhs</div>
        <div class="foot">
            Multiple Linear Regression | INR
        </div>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.metric("Property Area", f"{area:,} sq. ft.")
    st.metric(
        "Bedrooms / Bathrooms",
        f"{bedrooms} / {bathrooms}"
    )
    st.metric("Property Age", f"{age} years")
    st.metric("Parking Spaces", f"{parking}")

# ==========================================================
# 8. MODEL PERFORMANCE
# ==========================================================
st.markdown("")
st.markdown(
    '<div class="section-label">02 / MODEL HEALTH</div>',
    unsafe_allow_html=True
)
st.markdown("## Model Performance")

m1, m2, m3, m4 = st.columns(4)

m1.metric("MAE", f"₹ {mae:.2f} L")
m2.metric("MSE", f"{mse:.2f}")
m3.metric("RMSE", f"₹ {rmse:.2f} L")
m4.metric("R² Score", f"{r2:.3f}")

with st.expander("Understand the performance metrics"):
    st.markdown("""
    - **MAE:** Average absolute prediction error.
    - **MSE:** Average squared prediction error.
    - **RMSE:** Square root of MSE, in price units.
    - **R²:** Indicates how much test-price variation is
      explained by the model.
    """)

# ==========================================================
# 9. DATASET EXPLORER
# ==========================================================
st.markdown("")
st.markdown(
    '<div class="section-label">03 / DATA EXPLORER</div>',
    unsafe_allow_html=True
)
st.markdown("## Explore the Dataset")

d1, d2, d3 = st.columns(3)
d1.metric("Total Properties", len(df))
d2.metric("Input Features", len(FEATURES))
d3.metric(
    "Average Price",
    f"₹ {df['Price_lakhs'].mean():.1f} L"
)

with st.expander("View dataset records", expanded=True):
    st.dataframe(
        df, hide_index=True, use_container_width=True
    )
    st.caption(
        f"Dataset size: {df.shape[0]} rows × "
        f"{df.shape[1]} columns"
    )

    st.download_button(
        "⬇ Download Dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="estateiq_house_prices.csv",
        mime="text/csv"
    )

# ==========================================================
# 10. CHART STYLE FUNCTION
# ==========================================================
def style_axes(ax):
    ax.set_facecolor("#FFFFFF")
    ax.grid(alpha=0.22, color="#94A3B8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors="#475467")
    ax.xaxis.label.set_color("#344054")
    ax.yaxis.label.set_color("#344054")
    ax.title.set_color("#101A2C")

# ==========================================================
# 11. VISUAL ANALYTICS
# ==========================================================
st.markdown("")
st.markdown(
    '<div class="section-label">04 / VISUAL ANALYTICS</div>',
    unsafe_allow_html=True
)
st.markdown("## Explore the Model")

tab1, tab2, tab3 = st.tabs([
    "📍 Area vs. Price",
    "🎯 Actual vs. Predicted",
    "🧭 Feature Coefficients"
])

# ----------------------------------------------------------
# TAB 1: AREA VS PRICE
# ----------------------------------------------------------
with tab1:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.patch.set_facecolor("#FFFFFF")

    ax.scatter(
        df["Area_sqft"],
        df["Price_lakhs"],
        s=95,
        color="#0F9D91",
        edgecolor="#FFFFFF",
        linewidth=1.4,
        label="Sample properties",
        zorder=3
    )

    slope, intercept = np.polyfit(
        df["Area_sqft"],
        df["Price_lakhs"],
        1
    )

    x_values = np.linspace(
        df["Area_sqft"].min(),
        df["Area_sqft"].max(),
        100
    )

    ax.plot(
        x_values,
        slope * x_values + intercept,
        color="#E8B45B",
        linewidth=2.5,
        linestyle="--",
        label="Area-only trend"
    )

    ax.set_title(
        "Property Size and Listed Price",
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=14
    )
    ax.set_xlabel("Built-up area (square feet)")
    ax.set_ylabel("Price (₹ lakhs)")
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "The dashed line shows the area-only trend, not "
        "the full multiple linear regression model."
    )

# ----------------------------------------------------------
# TAB 2: ACTUAL VS PREDICTED
# ----------------------------------------------------------
with tab2:
    comparison = pd.DataFrame({
        "Actual Price (₹ lakhs)": y_test.to_numpy(),
        "Predicted Price (₹ lakhs)": y_pred
    })

    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.patch.set_facecolor("#FFFFFF")

    ax.scatter(
        comparison["Actual Price (₹ lakhs)"],
        comparison["Predicted Price (₹ lakhs)"],
        s=110,
        color="#0F9D91",
        edgecolor="#FFFFFF",
        linewidth=1.5,
        zorder=3
    )

    low = min(
        comparison["Actual Price (₹ lakhs)"].min(),
        comparison["Predicted Price (₹ lakhs)"].min(),
        0
    )

    high = max(
        comparison["Actual Price (₹ lakhs)"].max(),
        comparison["Predicted Price (₹ lakhs)"].max(),
        1
    )

    ax.plot(
        [low, high],
        [low, high],
        color="#E8B45B",
        linewidth=2.3,
        linestyle="--",
        label="Perfect prediction"
    )

    ax.set_title(
        "Actual Prices vs. Predictions",
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=14
    )
    ax.set_xlabel("Actual price (₹ lakhs)")
    ax.set_ylabel("Predicted price (₹ lakhs)")
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.dataframe(
        comparison.round(2),
        hide_index=True,
        use_container_width=True
    )

# ----------------------------------------------------------
# TAB 3: FEATURE COEFFICIENTS
# ----------------------------------------------------------
with tab3:
    coefficient_df = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": model.coef_
    }).sort_values("Coefficient")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.patch.set_facecolor("#FFFFFF")

    bar_colors = [
        "#E8B45B" if value < 0 else "#0F9D91"
        for value in coefficient_df["Coefficient"]
    ]

    ax.barh(
        coefficient_df["Feature"],
        coefficient_df["Coefficient"],
        color=bar_colors,
        height=0.62
    )

    ax.axvline(0, color="#344054", linewidth=1)
    ax.set_title(
        "Estimated Feature Coefficients",
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=14
    )
    ax.set_xlabel("Coefficient value")
    ax.set_ylabel("")
    style_axes(ax)
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.dataframe(
        coefficient_df.round(4),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Coefficients estimate how the prediction changes "
        "when a feature increases by one unit, holding other "
        "features constant. Estimates can be unstable with "
        "only 10 records."
    )

# ==========================================================
# 12. MODEL EQUATION
# ==========================================================
st.markdown("")
st.markdown(
    '<div class="section-label">05 / MODEL TRANSPARENCY</div>',
    unsafe_allow_html=True
)
st.markdown("## Regression Equation")

equation_parts = [
    f"({coef:.4f} × {feature})"
    for feature, coef in zip(FEATURES, model.coef_)
]

equation = (
    f"Price (₹ lakhs) = {model.intercept_:.4f} + "
    " + ".join(equation_parts)
)

st.code(equation, language="text")

with st.expander("How the prediction works"):
    st.markdown("""
    1. The model learns relationships between house features and prices.
    2. The dataset is divided into training and testing records.
    3. Linear Regression learns coefficients from training data.
    4. The model predicts a price from the sidebar inputs.
    5. Evaluation metrics compare predictions with test prices.
    """)

# ==========================================================
# 13. FOOTER
# ==========================================================
st.divider()

st.markdown("""
<div class="footer">
    <b>ESTATEIQ</b> | House Price Intelligence Dashboard<br>
    Built with Streamlit, pandas, Matplotlib and scikit-learn<br>
    Educational demonstration project
</div>
""", unsafe_allow_html=True)
