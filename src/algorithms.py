"""
Farklı algoritmalar için skor hesaplama fonksiyonları
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


def malatya_centrality(df, base_nodes, new_node):
    """
    Malatya Centrality skoru hesaplar
    Bu algoritma, yeni düğümün base düğümlere olan mesafesine dayanır
    
    Args:
        df: Tüm veri DataFrame'i
        base_nodes: Base düğümlerin index listesi
        new_node: Yeni eklenen düğümün index'i
        
    Returns:
        Malatya Centrality skoru (numeric)
    """
    # Base düğümlerin özellik vektörleri
    base_features = df.iloc[base_nodes].values
    
    # Yeni düğümün özellik vektörü
    new_features = df.iloc[new_node].values.reshape(1, -1)
    
    # Özellikleri normalize et
    scaler = StandardScaler()
    base_scaled = scaler.fit_transform(base_features)
    new_scaled = scaler.transform(new_features)
    
    # Yeni düğümden base düğümlere olan Öklid mesafelerini hesapla
    distances = np.sqrt(np.sum((base_scaled - new_scaled) ** 2, axis=1))
    
    # Malatya Centrality: mesafelerin tersinin toplamı
    # Daha yakın düğümlere sahip olan düğümler daha yüksek skor alır
    centrality_score = np.sum(1 / (1 + distances))
    
    return float(centrality_score)


def kmeans_score(df, base_nodes, new_node, n_clusters=None):
    """
    K-Means algoritmasına göre skor hesaplar
    Yeni düğümün base kümelere uyumunu ölçer
    
    Args:
        df: Tüm veri DataFrame'i
        base_nodes: Base düğümlerin index listesi
        new_node: Yeni eklenen düğümün index'i
        n_clusters: Küme sayısı (None ise otomatik hesaplanır)
        
    Returns:
        K-Means skoru (numeric)
    """
    # Base düğümlerin özellik vektörleri
    base_features = df.iloc[base_nodes].values
    
    # Yeni düğümün özellik vektörü
    new_features = df.iloc[new_node].values.reshape(1, -1)
    
    # Özellikleri normalize et
    scaler = StandardScaler()
    base_scaled = scaler.fit_transform(base_features)
    new_scaled = scaler.transform(new_features)
    
    # Küme sayısını belirle
    if n_clusters is None:
        n_clusters = min(5, len(base_nodes) // 2) if len(base_nodes) > 2 else 2
    
    # K-Means modelini eğit
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(base_scaled)
    
    # Yeni düğümün en yakın küme merkezine uzaklığını hesapla
    distances_to_centers = np.sqrt(np.sum((kmeans.cluster_centers_ - new_scaled) ** 2, axis=1))
    min_distance = np.min(distances_to_centers)
    
    # Skor: uzaklığın tersi (daha yakın = daha yüksek skor)
    kmeans_score = 1 / (1 + min_distance)
    
    return float(kmeans_score)


def knn_score(df, base_nodes, new_node, k=None):
    """
    K-Nearest Neighbors algoritmasına göre skor hesaplar
    
    Args:
        df: Tüm veri DataFrame'i
        base_nodes: Base düğümlerin index listesi
        new_node: Yeni eklenen düğümün index'i
        k: Komşu sayısı (None ise otomatik hesaplanır)
        
    Returns:
        KNN skoru (numeric)
    """
    # Base düğümlerin özellik vektörleri
    base_features = df.iloc[base_nodes].values
    
    # Yeni düğümün özellik vektörü
    new_features = df.iloc[new_node].values.reshape(1, -1)
    
    # Özellikleri normalize et
    scaler = StandardScaler()
    base_scaled = scaler.fit_transform(base_features)
    new_scaled = scaler.transform(new_features)
    
    # k değerini belirle
    if k is None:
        k = min(5, len(base_nodes) // 2) if len(base_nodes) > 2 else 1
    
    # KNN modelini eğit
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(base_scaled)
    
    # En yakın k komşuyu bul
    distances, indices = knn.kneighbors(new_scaled)
    
    # Skor: komşu mesafelerinin ortalamasının tersi
    avg_distance = np.mean(distances[0])
    knn_score = 1 / (1 + avg_distance)
    
    return float(knn_score)


def random_forest_score(df, base_nodes, new_node):
    """
    Random Forest algoritmasına göre skor hesaplar
    Base düğümlerden bir hedef değişken tahmin eder ve yeni düğüm için skor üretir
    
    Args:
        df: Tüm veri DataFrame'i
        base_nodes: Base düğümlerin index listesi
        new_node: Yeni eklenen düğümün index'i
        
    Returns:
        Random Forest skoru (numeric)
    """
    # Base düğümlerin özellik vektörleri
    base_features = df.iloc[base_nodes].values
    
    # Yeni düğümün özellik vektörü
    new_features = df.iloc[new_node].values.reshape(1, -1)
    
    # Özellikleri normalize et
    scaler = StandardScaler()
    base_scaled = scaler.fit_transform(base_features)
    new_scaled = scaler.transform(new_features)
    
    # Hedef değişken olarak ilk özelliği kullan (veya popularity varsa onu)
    if 'popularity' in df.columns:
        target_idx = list(df.columns).index('popularity')
        y = df.iloc[base_nodes, target_idx].values
    else:
        # İlk özelliği hedef olarak kullan
        y = base_scaled[:, 0]
    
    # X: diğer tüm özellikler
    X_base = base_scaled
    X_new = new_scaled
    
    # Random Forest modelini eğit
    rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
    rf.fit(X_base, y)
    
    # Yeni düğüm için tahmin yap
    prediction = rf.predict(X_new)[0]
    
    # Skor: tahmin değeri (normalize edilmiş)
    rf_score = float(prediction)
    
    return rf_score


def naive_bayes_score(df, base_nodes, new_node):
    """
    Naive Bayes algoritmasına göre skor hesaplar
    
    Args:
        df: Tüm veri DataFrame'i
        base_nodes: Base düğümlerin index listesi
        new_node: Yeni eklenen düğümün index'i
        
    Returns:
        Naive Bayes skoru (numeric)
    """
    # Base düğümlerin özellik vektörleri
    base_features = df.iloc[base_nodes].values
    
    # Yeni düğümün özellik vektörü
    new_features = df.iloc[new_node].values.reshape(1, -1)
    
    # Özellikleri normalize et (Naive Bayes için gerekli)
    scaler = StandardScaler()
    base_scaled = scaler.fit_transform(base_features)
    new_scaled = scaler.transform(new_features)
    
    # Hedef değişken: popularity veya ilk özellik
    if 'popularity' in df.columns:
        target_idx = list(df.columns).index('popularity')
        y = df.iloc[base_nodes, target_idx].values
    else:
        # İlk özelliği hedef olarak kullan ve kategorilere ayır
        y_raw = base_scaled[:, 0]
        median_val = np.median(y_raw)
        y = (y_raw > median_val).astype(int)
    
    # Eğer y sürekli ise kategorilere ayır
    if y.dtype == float:
        median_val = np.median(y)
        y = (y > median_val).astype(int)
    
    # Naive Bayes modelini eğit
    nb = GaussianNB()
    nb.fit(base_scaled, y)
    
    # Yeni düğüm için olasılık tahmini
    probabilities = nb.predict_proba(new_scaled)[0]
    
    # Skor: maksimum olasılık
    nb_score = float(np.max(probabilities))
    
    return nb_score

