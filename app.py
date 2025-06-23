import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Data/test_ori.csv")

df = load_data()

# Sidebar
with st.sidebar:
    selected = option_menu(
        "Pilih Halaman",
        ["Overview", "EDA", "Visualisasi Kinerja", "Prediksi", "Feature Importance"]
    )

# Overview Page
if selected == "Overview":
    st.title("Airline Satisfaction Overview")
    
    st.metric("Total Penumpang", len(df))
    st.metric("Rata-rata Usia", f"{df['Age'].mean():.2f} tahun")

    pie_data = df['satisfaction'].value_counts().reset_index()
    pie_data.columns = ['satisfaction', 'count']
    fig_pie = px.pie(pie_data, names='satisfaction', values='count', title='Distribusi Kepuasan')
    st.plotly_chart(fig_pie)

    bar_data = df.groupby(['Gender', 'satisfaction']).size().reset_index(name='count')
    fig_bar = px.bar(bar_data, x='Gender', y='count', color='satisfaction', barmode='group')
    st.plotly_chart(fig_bar)

# EDA Page
elif selected == "EDA":
    st.title("Data Exploration")
    
    filter_col = st.selectbox("Filter Berdasarkan:", ["None", "Gender", "Class"])
    
    if filter_col != "None":
        filter_val = st.selectbox("Pilih:", df[filter_col].unique())
        df_filtered = df[df[filter_col] == filter_val]
    else:
        df_filtered = df.copy()

    st.subheader("Distribusi Umur")
    fig = px.histogram(df_filtered, x="Age", color="satisfaction")
    st.plotly_chart(fig)

    st.subheader("Boxplot: Inflight Wifi Service vs Kepuasan")
    fig2 = px.box(df_filtered, x='satisfaction', y='Inflight wifi service', color='satisfaction')
    st.plotly_chart(fig2)

# Visualisasi Kinerja
elif selected == "Visualisasi Kinerja":
    st.title("Visualisasi Kinerja Layanan")

    fitur = ['Inflight wifi service', 'Food and drink', 'Seat comfort',
             'Inflight entertainment', 'On-board service', 'Leg room service']
    
    for f in fitur:
        st.subheader(f)
        fig = px.histogram(df, x=f, color='satisfaction', barmode='group')
        st.plotly_chart(fig)

# Prediksi Page
elif selected == "Prediksi":
    st.title("Prediksi Kepuasan Penumpang")

    st.write("Masukkan informasi penumpang:")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Usia", 18, 85, 30)
    travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
    travel_type = st.selectbox("Type of Travel", ["Personal Travel", "Business travel"])
    wifi = st.slider("Inflight Wifi Service", 0, 5, 3)
    seat = st.slider("Seat Comfort", 0, 5, 3)
    entertainment = st.slider("Inflight Entertainment", 0, 5, 3)
    
    # Load model (atau load dari model.pkl)
    model_path = "data/model.pkl"
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    except:
        st.warning("Model belum tersedia, menggunakan model dummy sementara.")
        le_gender = LabelEncoder().fit(["Male", "Female"])
        le_class = LabelEncoder().fit(["Eco", "Eco Plus", "Business"])
        le_type = LabelEncoder().fit(["Personal Travel", "Business travel"])
        X = pd.DataFrame({
            'Gender': le_gender.transform(df['Gender']),
            'Age': df['Age'],
            'Class': le_class.transform(df['Class']),
            'Type of Travel': le_type.transform(df['Type of Travel']),
            'Inflight wifi service': df['Inflight wifi service'],
            'Seat comfort': df['Seat comfort'],
            'Inflight entertainment': df['Inflight entertainment']
        })
        y = (df['satisfaction'] == 'satisfied').astype(int)
        model = RandomForestClassifier().fit(X, y)
        with open(model_path, "wb") as file:
            pickle.dump(model, file)

    # Encoding input
    le_gender = LabelEncoder().fit(["Male", "Female"])
    le_class = LabelEncoder().fit(["Eco", "Eco Plus", "Business"])
    le_type = LabelEncoder().fit(["Personal Travel", "Business travel"])

    input_df = pd.DataFrame([{
        'Gender': le_gender.transform([gender])[0],
        'Age': age,
        'Class': le_class.transform([travel_class])[0],
        'Type of Travel': le_type.transform([travel_type])[0],
        'Inflight wifi service': wifi,
        'Seat comfort': seat,
        'Inflight entertainment': entertainment
    }])

    if st.button("Prediksi"):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][pred]
        label = "Puas" if pred == 1 else "Tidak Puas"
        st.success(f"Hasil: **{label}**  |  Confidence: **{prob*100:.2f}%**")
        
        # Visualisasi hasil prediksi
        st.subheader("Visualisasi Hasil Prediksi")
        
        # Gauge chart untuk confidence level
        fig = px.pie(values=[prob, 1-prob], 
                     names=["Confidence", "Uncertainty"],
                     hole=0.7, 
                     title=f"Confidence Level: {prob*100:.2f}%")
        fig.update_traces(marker=dict(colors=['#00cc96' if pred == 1 else '#ef553b', '#cccccc']))
        st.plotly_chart(fig)
        
        # Radar chart untuk input features
        feature_names = ['Wifi Service', 'Seat Comfort', 'Entertainment']
        feature_values = [wifi, seat, entertainment]
        fig_radar = px.line_polar(
            r=feature_values, 
            theta=feature_names, 
            line_close=True,
            range_r=[0, 5],
            title="Feature Values")
        fig_radar.update_traces(fill='toself')
        st.plotly_chart(fig_radar)
        
        # Compare dengan rata-rata berdasarkan satisfaction
        avg_satisfied = df[df['satisfaction'] == 'satisfied'][['Inflight wifi service', 'Seat comfort', 'Inflight entertainment']].mean()
        avg_dissatisfied = df[df['satisfaction'] == 'neutral or dissatisfied'][['Inflight wifi service', 'Seat comfort', 'Inflight entertainment']].mean()
        
        compare_data = pd.DataFrame({
            'Feature': ['Wifi Service', 'Seat Comfort', 'Entertainment'],
            'Input': [wifi, seat, entertainment],
            'Rata-rata Puas': avg_satisfied.values,
            'Rata-rata Tidak Puas': avg_dissatisfied.values
        })
        
        fig_bar = px.bar(compare_data, x='Feature', y=['Input', 'Rata-rata Puas', 'Rata-rata Tidak Puas'], 
                         barmode='group',
                         title="Perbandingan dengan Rata-rata")
        st.plotly_chart(fig_bar)

# Feature Importance
elif selected == "Feature Importance":
    st.title("Feature Importance (SHAP Style)")

    try:
        with open("data/model.pkl", "rb") as file:
            model = pickle.load(file)
    except:
        st.error("Model belum tersedia.")
        st.stop()

    # Nama-nama fitur yang digunakan dalam model
    feature_names = ['Gender', 'Age', 'Class', 'Type of Travel', 
                     'Inflight wifi service', 'Seat comfort', 'Inflight entertainment']
    importance = model.feature_importances_
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    imp_df = imp_df.sort_values(by="Importance", ascending=True)

    # Bar chart horizontal
    fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title="Feature Importance")
    st.plotly_chart(fig)

    # --- Tambahan Visualisasi 3D ---
    st.subheader("Visualisasi 3D dari Fitur Terpenting")

    # Ambil 3 fitur dengan importance tertinggi
    top_features = imp_df.sort_values(by="Importance", ascending=False).head(3)["Feature"].tolist()

    st.markdown(f"**Tiga fitur yang divisualisasikan:** `{top_features[0]}`, `{top_features[1]}`, `{top_features[2]}`")

    # Salin data dan encode fitur kategorikal jika perlu
    df_encoded = df.copy()
    mapping = {
        'Gender': {'Male': 0, 'Female': 1},
        'Class': {'Eco': 0, 'Eco Plus': 1, 'Business': 2},
        'Type of Travel': {'Personal Travel': 0, 'Business travel': 1}
    }
    for col in mapping:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping[col])

    # Buat scatter 3D
    try:
        fig_3d = px.scatter_3d(
            df_encoded,
            x=top_features[0],
            y=top_features[1],
            z=top_features[2],
            color='satisfaction',
            opacity=0.6,
            title="3D Scatter Plot berdasarkan Tiga Fitur Terpenting",
            color_discrete_map={
                'satisfied': '#00cc96',
                'neutral or dissatisfied': '#ef553b'
            }
        )
        st.plotly_chart(fig_3d)
    except Exception as e:
        st.warning(f"Gagal membuat visualisasi 3D: {e}")
