
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


#Page title
st.set_page_config(
    page_title="Credit Portfolio Segmentation and Risk Pattern Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'Credit Portfolio Risk Analysis Dashboard'
    }
)

#Loadind all the Results
DATA_DIR = Path("streamlit_data")

@st.cache_data
def load_results():

    files = {
        "preview": "dataset_preview.csv",
        "business": "business_summary.csv",
        "evaluation": "evaluation_metrics.csv",
        "kmeans": "kmeans_results.csv",
        "default": "default_profile.csv",
        "profile": "cluster_numeric_profile.csv",
        "projection": "projection_results.csv",
        "tsne": "tsne_results.csv",
        "anomalies": "anomaly_examples.csv"
    }

    results = {}

    for key, filename in files.items():
        results[key] = pd.read_csv(DATA_DIR / filename)

    return results

results = load_results()


#Sidebar
st.sidebar.title("Credit Portfolio Risk Pattern Analysis")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Borrower Segmentation",
        "Dimensionality Reduction",
        "Clustering Evaluation",
        "Default & Anomaly Analysis"
    ]
)

st.sidebar.divider()



#Overview page
if page == "Overview":

    st.title("Credit Portfolio Dashboard")

    st.write(
        "Explore borrower segments, loan characteristics "
        "and historical default patterns."
    )

    st.divider()

    business = results["business"]

    total_borrowers = business["borrower_count"].sum()

    total_clusters = business["cluster"].nunique()

    total_anomalies = business["anomaly_count"].sum()

    overall_default = (
        (business["historical_default_rate"] *
         business["borrower_count"]).sum()
        / total_borrowers
    ) * 100

    # KPI cards

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Borrowers", f"{total_borrowers:,.0f}")
    c2.metric("Clusters", f"{total_clusters}")
    c3.metric("Anomalies", f"{total_anomalies:,.0f}")
    c4.metric("Historical Default", f"{overall_default:.2f}%")

    st.divider()

    st.subheader("Borrower Dataset Preview")

    st.dataframe(
        results["preview"].head(10),
        use_container_width=True
    )

    st.subheader("Portfolio Summary")

    st.dataframe(
        business.round(2),
        use_container_width=True
    )

#Borrowers Segmentation

elif page == "Borrower Segmentation":

    st.title("Borrower Segmentation")

    business = results["business"]

    st.write(
        "Understand the different borrower groups "
        "identified using K-Means clustering."
    )

    st.subheader("Borrower Count by Cluster")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        business["cluster"].astype(str),
        business["borrower_count"]
    )

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Borrowers")
    ax.set_title("Borrower Distribution")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Cluster Characteristics")

    st.dataframe(
        business.round(2),
        use_container_width=True
    )

    st.subheader("Median Financial Characteristics")

    st.dataframe(
        results["profile"],
        use_container_width=True
    )



#Dimentionality Reduction
elif page == "Dimensionality Reduction":

    st.title("Dimensionality Reduction")

    st.write(
        "Explore borrower data in two-dimensional space."
    )

    method = st.selectbox(
        "Select visualization method",
        ["PCA", "t-SNE", "UMAP"]
    )

    if method == "PCA":

        plot_data = results["projection"]

        x_col = "PCA_1"
        y_col = "PCA_2"

    elif method == "UMAP":

        plot_data = results["projection"]

        x_col = "UMAP_1"
        y_col = "UMAP_2"

    else:

        plot_data = results["tsne"]

        x_col = "TSNE_1"
        y_col = "TSNE_2"

    fig, ax = plt.subplots(figsize=(8, 5))

    scatter = ax.scatter(
        plot_data[x_col],
        plot_data[y_col],
        c=plot_data["Cluster"],
        cmap="viridis",
        s=8,
        alpha=0.6
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{method} Borrower Visualization")

    fig.colorbar(scatter, ax=ax, label="Cluster")

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Each point represents a borrower. Colors represent "
        "the K-Means cluster assignment."
    )



#Clustering Evaluation
elif page == "Clustering Evaluation":

    st.title("Clustering Evaluation")

    st.write(
        "Compare the clustering algorithms using "
        "their evaluation metrics."
    )

    evaluation = results["evaluation"]

    st.subheader("Evaluation Metrics")

    st.dataframe(
        evaluation.round(4),
        use_container_width=True
    )

    st.subheader("K-Means Cluster Selection")

    kmeans = results["kmeans"]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        kmeans["k"],
        kmeans["silhouette"],
        marker="o"
    )

    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("K-Means Silhouette Comparison")

    st.pyplot(fig)
    plt.close(fig)



#Default and Anomaly Detection
elif page == "Default & Anomaly Analysis":

    st.title("Default & Anomaly Analysis")

    default_data = results["default"]

    st.subheader("Historical Default Rate by Cluster")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        default_data["cluster"].astype(str),
        default_data["historical_default_rate_pct"]
    )

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Historical Default Rate (%)")
    ax.set_title("Default Rate Across Borrower Segments")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Default Profile")

    st.dataframe(
        default_data.round(2),
        use_container_width=True
    )

    st.divider()

    st.subheader("Isolation Forest Anomaly Examples")

    st.write(
        "Examples of observations identified as anomalies "
        "by the Isolation Forest model."
    )

    st.dataframe(
        results["anomalies"],
        use_container_width=True
    )


# ==========================================
# 9. FOOTER
# ==========================================

st.divider()

st.caption(
    "Credit Portfolio Segmentation and Risk Pattern Analysis | "
    "Machine Learning Dashboard"
)


