# app.py -- Minimal AI Data Insights (CSV only) - guaranteed to build
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="AI Data Insights (CSV)", layout="wide")
st.title("AI Data Insights Generator — CSV Only (Stable Build)")

st.write("Upload a CSV file (tables only). This lightweight version avoids heavy dependencies so it deploys reliably.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("No CSV uploaded — a small sample dataset will be used.")
    df = pd.DataFrame({
        "area": [800, 900, 1200, 1500, 1700, 2000],
        "bedrooms": [2, 2, 3, 3, 4, 4],
        "age": [10, 8, 5, 2, 1, 0],
        "price": [50, 55, 70, 90, 110, 130]
    })
else:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error("Failed to read CSV: " + str(e))
        st.stop()

# sanitize
df = df.dropna(axis=1, how="all")
st.subheader("Data preview")
st.write("Rows:", df.shape[0], "| Columns:", df.shape[1])
st.dataframe(df.head())

numeric_cols = df.select_dtypes(include="number").columns.tolist()
st.caption(f"Detected numeric columns: {numeric_cols}")

# quick insights
st.subheader("Quick automatic insights")
missing = df.isna().sum()
for c, m in missing.items():
    if m > 0:
        pct = m / max(len(df),1) * 100
        st.write(f"- Column **{c}** has {m} missing values ({pct:.1f}%).")

if len(numeric_cols) >= 2:
    st.subheader("Correlations (numeric)")
    corr = df[numeric_cols].corr()
    st.dataframe(corr.round(3))
    corrs = corr.abs().unstack().sort_values(ascending=False)
    corrs = corrs[corrs != 1.0]
    if not corrs.empty:
        top = corrs.index[0]
        st.write(f"Top correlation: **{top[0]}** ↔ **{top[1]}** = {corr.loc[top]:.2f}")
else:
    st.info("Not enough numeric columns for correlation analysis.")

# regression
st.subheader("Simple AI: Linear Regression")
target = st.selectbox("Select numeric target", [None] + numeric_cols)
features = st.multiselect("Select numeric features", numeric_cols)

if target and features:
    X = df[features].dropna()
    y = df.loc[X.index, target]
    if len(X) < 5:
        st.warning("Not enough rows for modeling (need at least 5).")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        model = LinearRegression().fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        st.write("R²:", round(r2,3), "| MAE:", round(mae,3))
        coef = dict(zip(features, model.coef_))
        st.write("Model coefficients:")
        st.table(pd.DataFrame.from_dict(coef, orient='index', columns=['coef']).round(3))
        compare = pd.DataFrame({"Actual": y_test, "Predicted": preds}).reset_index(drop=True)
        st.write(compare.head(10))

st.markdown("---")
st.caption("This minimal version was created so the app can be deployed immediately and reliably.")
