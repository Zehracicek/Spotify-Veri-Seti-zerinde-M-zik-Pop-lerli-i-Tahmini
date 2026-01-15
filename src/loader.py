"""
Spotify veri setini yükleme ve temizleme modülü
"""
import pandas as pd
import numpy as np
import os


def load_spotify_data(file_path='data/spotify_data.csv'):
    """
    Spotify CSV dosyasını yükler ve temizler
    
    Args:
        file_path: CSV dosyasının yolu
        
    Returns:
        Temizlenmiş DataFrame (sadece numeric özellikler)
    """
    # CSV dosyasını oku
    print(f"Veri yükleniyor: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"Orijinal veri boyutu: {df.shape}")
    
    # Numeric özellikleri seç
    # Spotify veri setindeki numeric kolonlar
    numeric_columns = [
        'popularity', 'year', 'danceability', 'energy', 'key', 
        'loudness', 'mode', 'speechiness', 'acousticness', 
        'instrumentalness', 'liveness', 'valence', 'tempo', 
        'duration_ms', 'time_signature'
    ]
    
    # Sadece mevcut numeric kolonları seç
    available_numeric = [col for col in numeric_columns if col in df.columns]
    
    # Numeric özellikleri içeren DataFrame oluştur
    df_numeric = df[available_numeric].copy()
    
    # Eksik değerleri kontrol et ve doldur
    missing_counts = df_numeric.isnull().sum()
    if missing_counts.sum() > 0:
        print("\nEksik değerler:")
        print(missing_counts[missing_counts > 0])
        
        # Numeric kolonlar için ortalama ile doldur
        for col in df_numeric.columns:
            if df_numeric[col].isnull().sum() > 0:
                df_numeric[col].fillna(df_numeric[col].mean(), inplace=True)
                print(f"{col} kolonu ortalama ile dolduruldu")
    
    # Sonsuz değerleri kontrol et ve temizle
    inf_counts = np.isinf(df_numeric.select_dtypes(include=[np.number])).sum()
    if inf_counts.sum() > 0:
        print("\nSonsuz değerler tespit edildi, temizleniyor...")
        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
        df_numeric = df_numeric.fillna(df_numeric.mean())
    
    # Index'i sıfırla
    df_numeric = df_numeric.reset_index(drop=True)
    
    print(f"Temizlenmiş veri boyutu: {df_numeric.shape}")
    print(f"Kullanılan numeric özellikler: {list(df_numeric.columns)}")
    
    return df_numeric


def get_data_subset(df, size):
    """
    Veri setinden belirli boyutta alt küme döndürür
    
    Args:
        df: DataFrame
        size: İstenen veri boyutu
        
    Returns:
        Alt küme DataFrame
    """
    if size > len(df):
        print(f"Uyarı: İstenen boyut ({size}) veri setinden büyük. Tüm veri kullanılacak.")
        return df.copy()
    
    return df.sample(n=size, random_state=42).reset_index(drop=True)

