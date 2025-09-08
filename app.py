import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Streamlit Starter: Explore, Visualize, Model",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Helpers (cached where sensible)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    # Try utf-8 then fallback to latin-1
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="latin-1")

@st.cache_data(show_spinner=False)
def profile_dataframe(df: pd.DataFrame) -> dict:
    info = {}
    info["shape"] = df.shape
    info["dtypes"] = df.dtypes.astype(str).to_dict()
    info["missing_per_column"] = df.isna().sum().to_dict()
    desc = df.describe(include="all").transpose()
    return {"info": info, "describe": desc}

@st.cache_data(show_spinner=False)
def generate_sample_df(n_rows=300, random_state=42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df = pd.DataFrame({
        "feature_1": rng.normal(0, 1, n_rows),
        "feature_2": rng.uniform(0, 10, n_rows),
        "category": rng.choice(["A", "B", "C"], n_rows, p=[0.4, 0.4, 0.2])
    })
    # Create a target with some relationship + noise
    df["target"] = 2.5 * df["feature_1"] - 0.7 * df["feature_2"] + \
                   (df["category"].map({"A": 1, "B": 0, "C": -1}).astype(float)) + \
                   rng.normal(0, 1, n_rows)
    return df

@st.cache_data(show_spinner=False)
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

@st.cache_data(show_spinner=False)
def train_regression(df: pd.DataFrame, target: str, features: list, test_size: float, random_state: int):
    X = df[features]
    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Separate numeric and categorical
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Preprocess
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
        # Keeping model simple; you could add OneHotEncoder here if desired.
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop"
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", LinearRegression())
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
    }

    results = {
        "y_test": y_test.reset_index(drop=True),
        "y_pred": pd.Series(y_pred, name="prediction")
    }
    return metrics, results

# -----------------------------
# State
# -----------------------------
if "df" not in st.session_state:
    st.session_state.df = None

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📁 Navigation")
page = st.sidebar.radio(
    "Go to",
    options=["Home", "Data Explorer", "Plotter", "Model (Regression)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Use the sample data if you don't have a CSV handy.")

# -----------------------------
# Pages
# -----------------------------
if page == "Home":
    st.title("📊 Streamlit Starter: Explore, Visualize, and Model")
    st.markdown("""
This app helps you quickly:
- **Upload** a CSV and explore data (preview, summary, missing values)
- **Visualize** with interactive Altair charts
- **Train** a simple regression model and view metrics

Use the sidebar to navigate.

If you want a custom version (classification, NLP, multipage layout, auth, database, etc.), let me know.
    """)

elif page == "Data Explorer":
    st.title("🧭 Data Explorer")

    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        use_sample = st.button("Or load sample data")

    if uploaded is not None:
        df = load_csv(uploaded)
        st.session_state.df = df
        st.success(f"Loaded: {uploaded.name} | Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    elif use_sample:
        df = generate_sample_df()
        st.session_state.df = df
        st.info(f"Loaded sample dataset | Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    df = st.session_state.df
    if df is None:
        st.warning("Upload a CSV or click 'Or load sample data' to begin.")
        st.stop()

    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)

    prof = profile_dataframe(df)
    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", prof["info"]["shape"][0])
    with c2:
        st.metric("Columns", prof["info"]["shape"][1])
    with c3:
        st.metric("Missing (total)", int(df.isna().sum().sum()))

    with st.expander("Column dtypes"):
        st.json(prof["info"]["dtypes"])
    with st.expander("Missing per column"):
        st.json(prof["info"]["missing_per_column"])
    with st.expander("Describe (numeric + categorical)"):
        st.dataframe(prof["describe"], use_container_width=True)

    st.subheader("Download")
    st.download_button(
        label="Download current data as CSV",
        data=to_csv_bytes(df),
        file_name="data_export.csv",
        mime="text/csv"
    )

elif page == "Plotter":
    st.title("📈 Plotter (Altair)")

    df = st.session_state.df
    if df is None:
        st.warning("No data found. Go to 'Data Explorer' to upload or load sample data.")
        st.stop()

    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in cols if c not in numeric_cols]

    st.markdown("Configure your chart:")
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
    with c1:
        chart_type = st.selectbox("Chart type", ["Scatter", "Line", "Bar"], index=0)
    with c2:
        x = st.selectbox("X axis", options=cols)
    with c3:
        y = st.selectbox("Y axis", options=cols)
    with c4:
        color = st.selectbox("Color (optional)", options=["(none)"] + cols, index=0)

    color_enc = None if color == "(none)" else color

    tooltip = [x, y] + ([color_enc] if color_enc else [])
    base = alt.Chart(df).mark_point().encode(
        x=alt.X(x, sort=None),
        y=alt.Y(y, sort=None),
        tooltip=tooltip
    )

    if chart_type == "Scatter":
        chart = base.mark_point(opacity=0.7)
    elif chart_type == "Line":
        chart = alt.Chart(df).mark_line().encode(
            x=alt.X(x, sort=None),
            y=alt.Y(y, sort=None),
            tooltip=tooltip
        )
    else:  # Bar
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(x, sort=None),
            y=alt.Y(y, sort=None),
            tooltip=tooltip
        )

    if color_enc:
        chart = chart.encode(color=color_enc)

    st.altair_chart(chart.properties(height=500).interactive(), use_container_width=True)

elif page == "Model (Regression)":
    st.title("🤖 Simple Regression Model")

    df = st.session_state.df
    if df is None:
        st.warning("No data found. Go to 'Data Explorer' to upload or load sample data.")
        st.stop()

    if df.shape[1] < 2:
        st.error("Need at least 2 columns (1 target + 1 feature).")
        st.stop()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    st.markdown("Select your target and features:")
    c1, c2 = st.columns([1, 2])
    with c1:
        target = st.selectbox("Target (numeric recommended)", options=all_cols,
                              index=all_cols.index("target") if "target" in all_cols else 0)
    with c2:
        features = st.multiselect(
            "Features (exclude target)",
            options=[c for c in all_cols if c != target],
            default=[c for c in numeric_cols if c != target][:2]
        )

    c3, c4, c5 = st.columns(3)
    with c3:
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    with c4:
        random_state = st.number_input("Random state", value=42, step=1)
    with c5:
        train_btn = st.button("Train model")

    if train_btn:
        if len(features) == 0:
            st.error("Please choose at least one feature.")
            st.stop()

        with st.spinner("Training model..."):
            metrics, results = train_regression(df, target, features, test_size, int(random_state))

        st.subheader("Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("R²", f"{metrics['R2']:.4f}")
        m2.metric("MAE", f"{metrics['MAE']:.4f}")
        m3.metric("RMSE", f"{metrics['RMSE']:.4f}")

        st.subheader("Predicted vs Actual (Test set)")
        result_df = pd.DataFrame({
            "actual": results["y_test"],
            "prediction": results["y_pred"]
        })
        scatter = alt.Chart(result_df).mark_point(opacity=0.7).encode(
            x="actual:Q", y="prediction:Q", tooltip=["actual", "prediction"]
        )
        line = alt.Chart(pd.DataFrame({"actual": result_df["actual"]}))
        line = alt.Chart(pd.DataFrame({"a": [result_df["actual"].min(), result_df["actual"].max()]})).mark_line(
        ).encode(x="a:Q", y="a:Q")
        st.altair_chart((scatter + line).properties(height=500).interactive(), use_container_width=True)

        st.subheader("Predictions (Top 50)")
        st.dataframe(result_df.head(50), use_container_width=True)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit. Ask for customizations anytime.")
