import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Konfigurasi halaman
st.set_page_config(
    page_title="Clustering Customer E-Commerce",
    page_icon="🛒",
    layout="wide"
)


st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #00b4db;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>🛒 Aplikasi Clustering Berbasis Web</h1>
        <p>Untuk Identifikasi Pola Pembelian dan Segmentasi Customer E-Commerce</p>
    </div>
""", unsafe_allow_html=True)

# Informasi metode
with st.expander("ℹ️ Tentang Aplikasi & K-Means Clustering", expanded=False):
    st.markdown("""
    **Aplikasi Clustering Customer E-Commerce** menggunakan algoritma K-Means untuk mengelompokkan pelanggan 
    berdasarkan pola pembelian dan karakteristik mereka.
    
    **Manfaat untuk E-Commerce:**
    - 🎯 **Segmentasi Customer:** Kelompokkan pelanggan berdasarkan perilaku
    - 💰 **Strategi Marketing:** Target promosi yang tepat untuk setiap segmen
    - 📊 **Analisis Pola:** Identifikasi customer high-value vs low-value
    - 🎁 **Personalisasi:** Rekomendasi produk sesuai segmen
    - 📈 **Optimasi Revenue:** Fokus resource pada segmen profitable
    
    **Metode K-Means Clustering:**
    1. Tentukan jumlah cluster (K) - misal: Premium, Regular, Occasional
    2. Algoritma mengelompokkan customer dengan karakteristik serupa
    3. Setiap cluster memiliki centroid (titik pusat)
    4. Customer di-assign ke cluster terdekat
    
    **Contoh Segmentasi:**
    - 🏆 **Premium Customers:** High spending, frequent purchases
    - 💎 **Regular Customers:** Medium spending, loyal
    - 🌱 **New/Occasional:** Low spending, need nurturing
    """)

st.markdown("---")

# Upload file
st.markdown("### 📁 Upload Data Customer")
uploaded_file = st.file_uploader(
    "Pilih file CSV yang berisi data customer dan pola pembelian",
    type=["csv"],
    help="Format: Kolom pertama = ID/Nama Customer, kolom selanjutnya = fitur numerik (Usia, Income, Frekuensi Belanja, dll)"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Tampilkan dataset
    st.markdown("### 📋 Data Customer Anda")
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Total Customer", len(df), help="Jumlah customer dalam dataset")
    with col_info2:
        st.metric("Total Kolom", len(df.columns), help="Jumlah kolom/fitur")
    with col_info3:
        st.metric("Fitur Tersedia", len(df.columns) - 1, help="Fitur untuk clustering")
    
    st.dataframe(df, use_container_width=True)

    # Identifikasi kolom
    cols = df.columns.tolist()
    identifier_col = cols[0]
    feature_cols = cols[1:]

    st.markdown("---")

    # Pengaturan clustering dalam 2 kolom
    st.markdown("### ⚙️ Konfigurasi Clustering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎲 Pilih Fitur Customer")
        selected_features = st.multiselect(
            "Fitur untuk segmentasi customer",
            feature_cols,
            default=feature_cols,
            help="Pilih fitur seperti: Usia, Penghasilan, Frekuensi Belanja, Total Pembelian, dll"
        )
        
        normalize = st.checkbox(
            "Normalisasi Data (Direkomendasikan)",
            value=True,
            help="Standarisasi agar fitur dengan skala berbeda (misal: Usia vs Income) memiliki bobot seimbang"
        )

    with col2:
        st.markdown("#### 🎯 Parameter Clustering")
        n_clusters = st.slider(
            "Jumlah Segmen Customer (K)",
            min_value=2,
            max_value=10,
            value=3,
            help="Contoh: 3 = Premium, Regular, Occasional | 4 = VIP, Active, Passive, Churned"
        )
        
        random_state = st.number_input(
            "Random State",
            min_value=0,
            max_value=100,
            value=42,
            help="Untuk reprodusibilitas hasil"
        )
        
        max_iter = st.number_input(
            "Max Iterations",
            min_value=100,
            max_value=1000,
            value=300,
            step=100,
            help="Jumlah iterasi maksimal algoritma"
        )

    if len(selected_features) < 1:
        st.warning("⚠️ Pilih minimal 1 fitur untuk melakukan clustering.")
    else:
        st.markdown("---")
        
        # Cek missing values terlebih dahulu
        X_check = df[selected_features].copy()
        missing_count = X_check.isnull().sum().sum()
        
        # Jika ada missing values, tampilkan opsi handling
        if missing_count > 0:
            st.markdown("""
                <div class="warning-box">
                    <h4>⚠️ Missing Values Terdeteksi</h4>
                    <p>Ditemukan <b>{}</b> nilai kosong dalam data yang dipilih.</p>
                </div>
            """.format(missing_count), unsafe_allow_html=True)
            
            # Tampilkan detail missing values per kolom
            missing_info = X_check.isnull().sum()
            missing_info = missing_info[missing_info > 0]
            
            st.markdown("**Detail Missing Values:**")
            col_miss1, col_miss2 = st.columns(2)
            with col_miss1:
                for idx, (col, count) in enumerate(missing_info.items()):
                    if idx % 2 == 0:
                        st.write(f"• **{col}:** {count} nilai kosong ({count/len(df)*100:.1f}%)")
            with col_miss2:
                for idx, (col, count) in enumerate(missing_info.items()):
                    if idx % 2 == 1:
                        st.write(f"• **{col}:** {count} nilai kosong ({count/len(df)*100:.1f}%)")
            
            st.markdown("---")
            st.markdown("**Pilih Metode Penanganan Missing Values:**")
            
            handle_method = st.radio(
                "Metode Handling",
                ["Isi dengan Rata-rata (Mean)", "Isi dengan Median", "Hapus Baris dengan Missing Values"],
                help="Pilih metode untuk menangani nilai kosong sebelum clustering",
                key="missing_handler"
            )
            
            # Info metode
            if handle_method == "Isi dengan Rata-rata (Mean)":
                st.info("📊 **Mean Imputation:** Nilai kosong akan diisi dengan rata-rata kolom. Cocok untuk data yang terdistribusi normal.")
            elif handle_method == "Isi dengan Median":
                st.info("📊 **Median Imputation:** Nilai kosong akan diisi dengan nilai tengah. Lebih robust terhadap outlier.")
            else:
                st.info("📊 **Drop Rows:** Baris yang memiliki nilai kosong akan dihapus. Pilih ini jika missing values < 5% dari total data.")
        
        # Tombol clustering
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            cluster_btn = st.button("🚀 Jalankan Clustering", use_container_width=True)
        
        if cluster_btn:
            with st.spinner("⏳ Sedang memproses data dan melakukan clustering..."):
                try:
                    # Persiapan data
                    X_original = df[selected_features].copy()
                    
                    # Handle missing values jika ada
                    if missing_count > 0:
                        if handle_method == "Isi dengan Rata-rata (Mean)":
                            X_clean = X_original.fillna(X_original.mean())
                            st.success(f"✅ {missing_count} missing values telah diisi dengan nilai rata-rata (mean).")
                        elif handle_method == "Isi dengan Median":
                            X_clean = X_original.fillna(X_original.median())
                            st.success(f"✅ {missing_count} missing values telah diisi dengan nilai median.")
                        else:  # Hapus baris
                            rows_before = len(X_original)
                            X_clean = X_original.dropna()
                            # Update dataframe juga
                            df = df.loc[X_clean.index].reset_index(drop=True)
                            X_clean = X_clean.reset_index(drop=True)
                            rows_deleted = rows_before - len(X_clean)
                            st.success(f"✅ {rows_deleted} baris dengan missing values telah dihapus. Sisa: {len(X_clean)} baris.")
                    else:
                        X_clean = X_original
                    
                    # Cek apakah masih ada NaN
                    if X_clean.isnull().sum().sum() > 0:
                        st.error("❌ Masih ada missing values setelah cleaning. Silakan coba metode lain.")
                        st.stop()
                    
                    # Convert to numpy array
                    X = X_clean.values.astype(float)
                    
                    # Cek apakah ada infinite values
                    if np.isinf(X).any():
                        st.error("❌ Data mengandung nilai infinite. Silakan periksa data Anda.")
                        st.stop()
                    
                    # Cek apakah cukup data
                    if len(X) < n_clusters:
                        st.error(f"❌ Data terlalu sedikit ({len(X)} baris) untuk {n_clusters} cluster. Minimal butuh {n_clusters} baris data.")
                        st.stop()
                    
                    # Normalisasi jika dipilih
                    if normalize:
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                    else:
                        X_scaled = X
                    
                    # K-Means Clustering
                    kmeans = KMeans(
                        n_clusters=n_clusters,
                        random_state=random_state,
                        max_iter=max_iter,
                        n_init=10
                    )
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    # Tambahkan hasil ke dataframe
                    df['Cluster'] = clusters
                    df['Cluster_Label'] = df['Cluster'].apply(lambda x: f"Cluster {x + 1}")
                    
                    # Hitung jarak ke centroid
                    distances = kmeans.transform(X_scaled)
                    df['Distance_to_Centroid'] = [distances[i][clusters[i]] for i in range(len(distances))]
                    
                    # Simpan ke session state
                    st.session_state['clustering_done'] = True
                    st.session_state['df_result'] = df
                    st.session_state['kmeans'] = kmeans
                    st.session_state['n_clusters'] = n_clusters
                    st.session_state['selected_features'] = selected_features
                    st.session_state['identifier_col'] = identifier_col
                    st.session_state['normalize'] = normalize
                    if normalize:
                        st.session_state['scaler'] = scaler
                    
                    st.success("✅ Clustering berhasil! Scroll ke bawah untuk melihat hasil.")
                    
                except ValueError as e:
                    st.error(f"❌ Error saat memproses data: {str(e)}")
                    st.info("💡 Tips: Pastikan kolom yang dipilih berisi nilai numerik (angka), bukan teks.")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan: {str(e)}")
                    st.info("💡 Silakan periksa format data Anda atau hubungi administrator.")
                    st.stop()

        # Tampilkan hasil jika sudah clustering
        if 'clustering_done' in st.session_state and st.session_state['clustering_done']:
            df = st.session_state['df_result']
            kmeans = st.session_state['kmeans']
            n_clusters = st.session_state['n_clusters']
            selected_features = st.session_state['selected_features']
            identifier_col = st.session_state['identifier_col']
            
            st.markdown("---")
            
            # Hasil clustering
            st.markdown("### 🏆 Hasil Segmentasi Customer")
            
            # Statistik cluster
            st.markdown("#### 📊 Distribusi Segmen Customer")
            cluster_stats = df['Cluster_Label'].value_counts().sort_index()
            
            cols_stats = st.columns(n_clusters)
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788']
            
            for idx, (cluster_name, count) in enumerate(cluster_stats.items()):
                with cols_stats[idx]:
                    percentage = (count / len(df)) * 100
                    st.markdown(f"""
                        <div class="metric-card" style="border-top: 4px solid {colors[idx]};">
                            <h3 style="color: {colors[idx]}; margin: 0;">Segmen {idx + 1}</h3>
                            <h2 style="margin: 0.5rem 0; color: #333;">{count} Customer</h2>
                            <p style="color: #666; font-size: 0.9rem;">{percentage:.1f}% dari total</p>
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # Visualisasi
            st.markdown("#### 📈 Visualisasi Segmentasi Customer")
            
            if len(selected_features) >= 2:
                # Scatter plot 2D
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for i in range(n_clusters):
                    cluster_data = df[df['Cluster'] == i]
                    ax.scatter(
                        cluster_data[selected_features[0]], 
                        cluster_data[selected_features[1]],
                        c=colors[i],
                        label=f'Segmen {i+1}',
                        alpha=0.6,
                        s=100,
                        edgecolors='black',
                        linewidth=0.5
                    )
                
                # Plot centroids
                if 'scaler' in st.session_state and st.session_state['normalize']:
                    scaler = st.session_state['scaler']
                    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
                else:
                    centroids = kmeans.cluster_centers_
                
                ax.scatter(
                    centroids[:, 0], 
                    centroids[:, 1],
                    c='red',
                    marker='X',
                    s=300,
                    edgecolors='black',
                    linewidth=2,
                    label='Centroids'
                )
                
                ax.set_xlabel(selected_features[0], fontsize=12, fontweight='bold')
                ax.set_ylabel(selected_features[1], fontsize=12, fontweight='bold')
                ax.set_title(f'Pola Customer: {selected_features[0]} vs {selected_features[1]}', 
                           fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Jika ada 3 fitur atau lebih, buat 3D plot
                if len(selected_features) >= 3:
                    fig_3d = plt.figure(figsize=(12, 8))
                    ax_3d = fig_3d.add_subplot(111, projection='3d')
                    
                    for i in range(n_clusters):
                        cluster_data = df[df['Cluster'] == i]
                        ax_3d.scatter(
                            cluster_data[selected_features[0]], 
                            cluster_data[selected_features[1]],
                            cluster_data[selected_features[2]],
                            c=colors[i],
                            label=f'Segmen {i+1}',
                            alpha=0.6,
                            s=100,
                            edgecolors='black',
                            linewidth=0.5
                        )
                    
                    # Plot centroids 3D
                    if len(centroids[0]) >= 3:
                        ax_3d.scatter(
                            centroids[:, 0], 
                            centroids[:, 1],
                            centroids[:, 2],
                            c='red',
                            marker='X',
                            s=300,
                            edgecolors='black',
                            linewidth=2,
                            label='Centroids'
                        )
                    
                    ax_3d.set_xlabel(selected_features[0], fontsize=10, fontweight='bold')
                    ax_3d.set_ylabel(selected_features[1], fontsize=10, fontweight='bold')
                    ax_3d.set_zlabel(selected_features[2], fontsize=10, fontweight='bold')
                    ax_3d.set_title('Visualisasi 3D Segmentasi Customer', fontsize=14, fontweight='bold')
                    ax_3d.legend()
                    st.pyplot(fig_3d)
            else:
                # Histogram untuk 1 fitur
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for i in range(n_clusters):
                    cluster_data = df[df['Cluster'] == i]
                    ax.hist(
                        cluster_data[selected_features[0]], 
                        alpha=0.6, 
                        label=f'Segmen {i+1}',
                        color=colors[i],
                        bins=20,
                        edgecolor='black'
                    )
                
                ax.set_xlabel(selected_features[0], fontsize=12, fontweight='bold')
                ax.set_ylabel('Frekuensi', fontsize=12, fontweight='bold')
                ax.set_title(f'Distribusi {selected_features[0]} per Segmen', 
                           fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                st.pyplot(fig)

            st.markdown("---")

            # Profil cluster
            st.markdown("#### 🔍 Profil Setiap Segmen Customer")
            
            for i in range(n_clusters):
                cluster_data = df[df['Cluster'] == i]
                with st.expander(f"📌 Segmen {i + 1} ({len(cluster_data)} customer)", expanded=True):
                    col_profile1, col_profile2 = st.columns(2)
                    
                    with col_profile1:
                        st.markdown("**Statistik Deskriptif:**")
                        st.dataframe(
                            cluster_data[selected_features].describe().round(2),
                            use_container_width=True
                        )
                    
                    with col_profile2:
                        st.markdown("**Karakteristik Segmen:**")
                        means = cluster_data[selected_features].mean()
                        for feature in selected_features:
                            st.metric(
                                f"Rata-rata {feature}",
                                f"{means[feature]:.2f}"
                            )
                        
                        # Tambahan: Rekomendasi strategi berdasarkan profil
                        st.markdown("**💡 Rekomendasi Strategi:**")
                        if i == 0:
                            st.info("🏆 Segmen ini bisa jadi **High-Value Customers**. Pertimbangkan: Program loyalitas VIP, Exclusive offers")
                        elif i == 1:
                            st.info("💎 Segmen ini bisa jadi **Regular Customers**. Pertimbangkan: Cashback programs, Bundle deals")
                        else:
                            st.info("🌱 Segmen ini bisa jadi **Potential Customers**. Pertimbangkan: Welcome discount, Referral programs")

            st.markdown("---")
            
            # Tabel lengkap hasil
            st.markdown("#### 📊 Tabel Lengkap Customer & Segmentasi")
            
            # Sorting options
            col_sort1, col_sort2 = st.columns([2, 3])
            with col_sort1:
                sort_by = st.selectbox(
                    "Urutkan berdasarkan",
                    ['Cluster', identifier_col, 'Distance_to_Centroid'] + selected_features
                )
            with col_sort2:
                sort_order = st.radio("Urutan", ['Ascending', 'Descending'], horizontal=True)
            
            df_display = df.sort_values(
                by=sort_by,
                ascending=(sort_order == 'Ascending')
            )
            
            st.dataframe(df_display, use_container_width=True)

            # Download hasil
            csv = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Hasil Segmentasi Customer (CSV)",
                data=csv,
                file_name="customer_segmentation_result.csv",
                mime="text/csv",
            )

            st.markdown(f"""
                <div class="success-box">
                    ✅ <b>Segmentasi selesai!</b> Customer berhasil dikelompokkan menjadi {n_clusters} segmen. 
                    Gunakan insight ini untuk strategi marketing yang lebih tertarget dan personal!
                </div>
            """, unsafe_allow_html=True)

            # Metrik evaluasi
            st.markdown("---")
            st.markdown("#### 📉 Metrik Evaluasi Clustering")
            
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            
            with col_metric1:
                inertia = kmeans.inertia_
                st.metric(
                    "Inertia (WCSS)",
                    f"{inertia:.2f}",
                    help="Semakin kecil semakin baik (data lebih compact dalam cluster)"
                )
            
            with col_metric2:
                n_iter = kmeans.n_iter_
                st.metric(
                    "Jumlah Iterasi",
                    n_iter,
                    help="Jumlah iterasi hingga algoritma konvergen"
                )
            
            with col_metric3:
                avg_distance = df['Distance_to_Centroid'].mean()
                st.metric(
                    "Rata-rata Jarak ke Centroid",
                    f"{avg_distance:.4f}",
                    help="Rata-rata jarak data ke centroid cluster-nya"
                )

else:
    st.info("👆 Silakan upload file CSV berisi data customer untuk memulai analisis segmentasi.")
    
    # Tips untuk user
    with st.expander("💡 Tips Penggunaan Aplikasi"):
        st.markdown("""
        **Format Data yang Dibutuhkan:**
        - Kolom pertama: ID atau Nama Customer
        - Kolom selanjutnya: Fitur numerik seperti:
          - Usia
          - Penghasilan/Income
          - Frekuensi Belanja (per bulan/tahun)
          - Total Pembelian (nilai rupiah)
          - Lama berlangganan
          - Rating/Review yang diberikan
          - Dan fitur lain yang relevan
        
        **Contoh Format CSV:**
        ```
        Customer_ID,Usia,Income,Freq_Belanja,Total_Pembelian
        CUST001,25,5000000,10,15000000
        CUST002,45,15000000,25,50000000
        ```
        
        **Rekomendasi:**
        - Minimal 30-50 customer untuk hasil yang baik
        - Gunakan 2-4 fitur yang paling relevan
        - Aktifkan normalisasi data
        - Mulai dengan 3-4 segmen
        
        **Handling Missing Values:**
        - Aplikasi otomatis mendeteksi nilai kosong
        - Pilih metode yang sesuai:
          - **Mean:** Untuk data normal
          - **Median:** Untuk data dengan outlier
          - **Drop Rows:** Jika missing < 5%
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><b>Aplikasi Clustering Berbasis Web</b></p>
        <p>Untuk Identifikasi Pola Pembelian dan Segmentasi Customer E-Commerce</p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem;">Powered by K-Means Algorithm | Scikit-learn & Matplotlib</p>
    </div>
""", unsafe_allow_html=True)