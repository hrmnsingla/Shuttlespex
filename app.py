
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression

from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(layout="wide", page_title="Smashlytics Elite Dashboard")

# ---------- LOAD DATA ----------
df = pd.read_csv("badminton_synthetic_cleaned_5000.csv")

# ---------- SIDEBAR ----------
st.sidebar.title("🏸 Smashlytics")
st.sidebar.caption("Data-Driven Sports Intelligence")

page = st.sidebar.radio("Navigate", [
    "📊 EDA & Overview",
    "🎯 Customer Segmentation",
    "🤖 Purchase Intent (Classification)",
    "🛒 Association Mining",
    "💰 Spending Prediction",
    "🎯 Strategy Engine",
    "🔮 New Customer Predictor"
])

# ---------- EDA ----------
if page == "📊 EDA & Overview":
    st.title("📊 Exploratory Data Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Users", len(df))
    col2.metric("Avg Engagement", round(df["EngagementScore"].mean(),2))
    col3.metric("Avg Spend Score", round(df["SpendingScore"].mean(),2))
    col4.metric("Conversion Rate", f"{round(df['InterestBinary'].mean()*100,2)}%")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Demographics","Behavior","Correlation"])

    with tab1:
        fig = px.pie(df, names="PlayingLevel", title="Player Distribution")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.histogram(df, x="CityTier", color="Gender", barmode="group")
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        fig3 = px.histogram(df, x="MonthlySpend", title="Spending Pattern")
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        corr = df.select_dtypes(include=np.number).corr()
        fig4 = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig4, use_container_width=True)

# ---------- SEGMENTATION ----------
elif page == "🎯 Customer Segmentation":
    st.title("🎯 Customer Segmentation")

    X = df[["EngagementScore","SpendingScore","SeriousnessIndex"]]

    inertia = []
    sil = []
    K = range(2,9)

    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, labels))

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.line(x=list(K), y=inertia, markers=True, title="Elbow Method")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.line(x=list(K), y=sil, markers=True, title="Silhouette Score")
        st.plotly_chart(fig2, use_container_width=True)

    optimal_k = K[np.argmax(sil)]
    st.success(f"Optimal K = {optimal_k}")

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)

    fig3 = px.scatter(df, x="EngagementScore", y="SpendingScore",
                      color=df["Cluster"].astype(str),
                      size="SeriousnessIndex")
    st.plotly_chart(fig3, use_container_width=True)

# ---------- CLASSIFICATION ----------
elif page == "🤖 Purchase Intent (Classification)":
    st.title("🤖 Conversion Prediction Model")

    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include="object").columns:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    X = df_encoded.drop(columns=["InterestBinary","InterestLevel"])
    y = df_encoded["InterestBinary"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    st.metric("Accuracy", round(accuracy_score(y_test,y_pred),2))
    st.metric("Precision", round(precision_score(y_test,y_pred),2))
    st.metric("Recall", round(recall_score(y_test,y_pred),2))
    st.metric("F1 Score", round(f1_score(y_test,y_pred),2))

    fpr, tpr, _ = roc_curve(y_test,y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr,y=tpr,name="ROC"))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],line=dict(dash="dash")))
    st.plotly_chart(fig)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(10)

    fig2 = px.bar(importance, x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig2)

# ---------- ASSOCIATION ----------
elif page == "🛒 Association Mining":
    st.title("🛒 Association Rules")

    basket = pd.get_dummies(df[["Item_1","Item_2"]].stack()).groupby(level=0).sum()
    frequent = apriori(basket, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent, metric="confidence", min_threshold=0.5)

    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]].sort_values(by="lift", ascending=False))

# ---------- REGRESSION ----------
elif page == "💰 Spending Prediction":
    st.title("💰 Spending Prediction")

    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include="object").columns:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    X = df_encoded.drop(columns=["SpendingScore"])
    y = df_encoded["SpendingScore"]

    model = LinearRegression()
    model.fit(X,y)

    pred = model.predict(X)

    fig = px.scatter(x=y, y=pred, labels={"x":"Actual","y":"Predicted"})
    st.plotly_chart(fig)

# ---------- STRATEGY ----------
elif page == "🎯 Strategy Engine":
    st.title("🎯 Prescriptive Strategy")

    st.info("High Engagement + High Spending → Premium Yonex + Analytics Subscription")
    st.info("Low Engagement → Entry Level Kits + Discounts")

# ---------- PREDICT ----------
elif page == "🔮 New Customer Predictor":
    st.title("🔮 Predict Customer")

    level = st.selectbox("Level", df["PlayingLevel"].unique())
    spend = st.selectbox("Spend", df["MonthlySpend"].unique())

    if st.button("Predict"):
        st.success("Likely High Value Customer → Target with Premium Bundle")
