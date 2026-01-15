import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Rapordaki görselleştirme standartlarını uygulamak için stil ayarı
sns.set_theme(style="whitegrid")

def malatya_centrality_graf_olustur(df, top_n=50):
    """Rapor Sayfa 11: En Yüksek MC Skoruna Sahip 50 Düğümün Graf Yapısı"""
    # Özellikler arası korelasyonu ağ kenarları olarak kullanıyoruz [cite: 87]
    corr_matrix = df.corr().abs()
    G = nx.from_numpy_array(corr_matrix.values)
    
    # En merkezi düğümleri görselleştirme [cite: 276]
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42) 
    nx.draw(G, pos, node_color='#1DB954', node_size=100, edge_color='lightgray', with_labels=False)
    plt.title(f"En Yüksek Malatya Centrality Skoruna Sahip {top_n} Düğümün Graf Yapısı")
    plt.show()

def pca_yerlesimi_ciz(df, mc_skorlari):
    """Rapor Sayfa 12: Malatya Centrality Graf Yapısı (PCA Yerleşimi)"""
    # Veriyi normalize et ve PCA uygula 
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(scaled_data)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=mc_skorlari, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Malatya Centrality Skoru')
    plt.xlabel('PCA Bileşen 1')
    plt.ylabel('PCA Bileşen 2')
    plt.title('İlk 5000 Veri İçin Malatya Centrality Graf Yapısı (PCA Yerleşimi)')
    plt.show()

def performans_karsilastirma_grafikleri(sonuc_verisi):
    """Rapor Sayfa 10: Algoritmaların Performans Karşılaştırması"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Sınıflandırma Doğruluğu [cite: 230]
    sns.lineplot(ax=axes[0, 0], data=sonuc_verisi, x='Veri Miktarı', y='Accuracy', hue='Algoritma', marker='o')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_title('Sınıflandırma Doğruluğu')

    # 2. F1 Skoru [cite: 247]
    sns.lineplot(ax=axes[0, 1], data=sonuc_verisi, x='Veri Miktarı', y='F1_Score', hue='Algoritma', marker='s')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_title('F1 Skoru Karşılaştırması')

    # 3. İşlem Süresi [cite: 237]
    sns.lineplot(ax=axes[1, 0], data=sonuc_verisi, x='Veri Miktarı', y='Time', hue='Algoritma', marker='^')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('İşlem Süresi Karşılaştırması')

    # 4. K-Means Performansı (Silhouette) [cite: 249]
    # Sadece K-Means verisini filtreleyerek gösterim
    plt.tight_layout()
    plt.show()
