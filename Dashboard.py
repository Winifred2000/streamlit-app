import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import plotly.express as px
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from prophet.plot import plot_plotly, plot_components_plotly

st.set_page_config(layout="wide")
st.title("Nottingham Crime Trends Dashboard")
st.markdown("Forecasting and Analysis of Crime Data in Nottinghamshire")

# Load cleaned data
crime_data = pd.read_csv("clean_crime_data.csv", parse_dates=['Month'])

# Compute and display metrics
total_crimes = len(crime_data)
total_crime_types = crime_data['CrimeDecoded'].nunique()
total_locations = crime_data['Location'].nunique()

st.markdown("### 🔍 Overview Summary")
col1, col2, col3 = st.columns(3)
col1.metric("📌 Total Crimes", f"{total_crimes:,}")
col2.metric("🧩 Crime Types", total_crime_types)
col3.metric("📍 Total Crime Locations", total_locations)

# Sidebar filters
st.sidebar.header("Filter Data")

# Dropdown to select a single crime type or all
crime_type_options = ['All Crimes'] + sorted(crime_data['CrimeDecoded'].unique().tolist())
selected_crime_type = st.sidebar.selectbox("Select Crime Type", crime_type_options)

# Date range picker
date_range = st.sidebar.date_input(
    "Select Date Range",
    [crime_data['Month'].min(), crime_data['Month'].max()]
)

# Validate and extract date range
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
else:
    st.error("Please select a valid date range.")
    st.stop()

# Filter the data using selection
if selected_crime_type == 'All Crimes':
    filtered_data = crime_data[
        crime_data['Month'].between(start_date, end_date)
    ]
else:
    filtered_data = crime_data[
        (crime_data['CrimeDecoded'] == selected_crime_type) &
        (crime_data['Month'].between(start_date, end_date))
    ]



# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "📈 Forecast", "🧠 Hotspot Prediction", "🗺️ Crime Map"])

with tab1:
    st.subheader("📊 Crime Type Distribution")
    crime_counts = filtered_data['CrimeDecoded'].value_counts().reset_index()
    crime_counts.columns = ['Crime Type', 'Count']
    fig = px.bar(crime_counts, x='Count', y='Crime Type', orientation='h')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Monthly Crime Trend")
    monthly = filtered_data.groupby('Month').size().reset_index(name='Count')
    fig2 = px.line(monthly, x='Month', y='Count')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📍 Top 10 Crime Locations")
    top_locs = filtered_data['Location'].value_counts().nlargest(10).reset_index()
    top_locs.columns = ['Location', 'Count']
    fig3 = px.bar(top_locs, x='Count', y='Location', orientation='h')
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📅 Heatmap of Crime Types by Month")
    heatmap_data = crime_data.groupby(['Month_Num', 'CrimeDecoded']).size().reset_index(name='Count')
    fig4 = px.density_heatmap(
        heatmap_data,
        x='Month_Num',
        y='CrimeDecoded',
        z='Count',
        color_continuous_scale='YlOrRd'
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("🗓️ Yearly Crime Type Heatmap")

    # Ensure YearMonth is datetime
    crime_data['YearMonth'] = pd.to_datetime(crime_data['YearMonth'], errors='coerce')
    if hasattr(crime_data['YearMonth'].dt, 'to_timestamp'):
        crime_data['YearMonth'] = crime_data['YearMonth'].dt.to_timestamp()

    crime_data['year'] = crime_data['YearMonth'].dt.year
    crime_data['month_num'] = crime_data['YearMonth'].dt.month

    # Year selector
    available_years = sorted(crime_data['year'].dropna().unique())
    selected_year = st.selectbox("Select Year to View Crime Heatmap", available_years, key="yearly_heatmap_selector")

    # Filter and pivot
    data_year = crime_data[crime_data['year'] == selected_year]
    pivot = data_year.groupby(['CrimeDecoded', 'month_num']).size().unstack(fill_value=0)

    # Plot heatmap
    fig5, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt="d", ax=ax)
    ax.set_title(f"Crime Heatmap for {selected_year}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Crime Type")
    plt.tight_layout()
    st.pyplot(fig5)


with tab2:
    st.subheader("Crime Forecast (All Types Combined)")

    monthly = filtered_data.groupby('Month').size().reset_index(name='y')
    monthly.rename(columns={'Month': 'ds'}, inplace=True)

    if len(monthly) >= 24:
        model = Prophet()
        model.fit(monthly)
        future = model.make_future_dataframe(periods=6, freq='ME')
        forecast = model.predict(future)


        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1, use_container_width=True)


        st.subheader("Forecast Components")
        fig2 = plot_components_plotly(model, forecast)
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("Not enough data to perform forecasting.")

with tab3:
    st.subheader("🧠 Hotspot Prediction Summary")

    st.markdown("""
    This section includes the classification model's results,
    such as confusion matrices and an interactive visualisation
    of predicted crime hotspots.
    """)

    # ========== HOTSPOT PREDICTION ==========

    # Group and engineer features
    loc_month = crime_data.groupby(['Location', 'YearMonth', 'Crime type', 'Cluster']).size().reset_index(name='Crime_Count')
    loc_month['Hotspot'] = (loc_month['Crime_Count'] > 20).astype(int)

    loc_month['YearMonth'] = pd.to_datetime(loc_month['YearMonth'])
    loc_month['Month'] = loc_month['YearMonth'].dt.month
    loc_month['Year'] = loc_month['YearMonth'].dt.year
    loc_month['Prev_Crimes'] = loc_month.groupby('Location')['Crime_Count'].shift(1)
    loc_month.dropna(inplace=True)

    # Encode categorical features
    le_crime = LabelEncoder()
    le_cluster = LabelEncoder()
    loc_month['CrimeTypeEncoded'] = le_crime.fit_transform(loc_month['Crime type'])
    loc_month['ClusterEncoded'] = le_cluster.fit_transform(loc_month['Cluster'].astype(str))

    # Model features and target
    features = ['Month', 'Year', 'Prev_Crimes', 'CrimeTypeEncoded', 'ClusterEncoded']
    X = loc_month[features]
    y = loc_month['Hotspot']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Generate confusion matrix
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    conf_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Plot interactive confusion matrix
    fig = px.imshow(
        conf_df,
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=conf_df.columns,
        y=conf_df.index,
        title="Interactive Confusion Matrix: Hotspot Prediction"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Also show static version if needed
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix: Hotspot Prediction")
    plt.tight_layout()

    output_path = os.path.join(os.getcwd(), "hotspot_confusion_matrix.png")
    plt.savefig(output_path)
    plt.close()

    st.image(output_path, caption="(Optional) Static Confusion Matrix PNG")



with tab4:
    st.subheader("Crime Density Heatmap")
    with open("crime_density_heatmap.html", "r", encoding="utf-8") as f:
        html_content = f.read()
        components.html(html_content, height=600, scrolling=True)

