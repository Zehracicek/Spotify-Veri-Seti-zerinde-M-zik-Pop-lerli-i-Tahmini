"""
Ana çalıştırma dosyası
Spotify veri seti üzerinde veri madenciliği deneyleri
"""
import os
import sys
import pandas as pd
from .loader import load_spotify_data, get_data_subset
from .experiments import run_all_experiments, create_comparison_table
from .plotting import create_all_plots


def main():
    """
    Ana fonksiyon: Tüm deneyleri çalıştırır ve sonuçları kaydeder
    """
    print("="*70)
    print("SPOTIFY VERİ MADENCİLİĞİ DENEYLERİ")
    print("="*70)
    
    # 1. Veriyi yükle
    print("\n[1/5] Veri yükleniyor...")
    # src/ klasöründen çalıştırıldığı için bir üst dizine çık
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'spotify_data.csv')
    df = load_spotify_data(data_path)
    
    # Veri boyutunu kontrol et
    max_data_size = len(df)
    print(f"Mevcut veri boyutu: {max_data_size}")
    
    # 2. Base boyut listesini tanımla
    base_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    
    # Veri boyutuna göre uygun base size'ları filtrele
    # En büyük base size için en az 2*base_size veri gerekli
    valid_base_sizes = [bs for bs in base_sizes if max_data_size >= bs * 2]
    
    if not valid_base_sizes:
        print("HATA: Veri boyutu yeterli değil. En az 20 satır gerekli.")
        return
    
    print(f"\nKullanılacak base boyutları: {valid_base_sizes}")
    
    # Eğer veri çok büyükse, performans için alt küme al
    # Maksimum 10000 satır kullan (deneyler için yeterli)
    if max_data_size > 10000:
        print(f"\nVeri boyutu çok büyük ({max_data_size}). Performans için 10000 satır kullanılacak.")
        df = get_data_subset(df, 10000)
        max_data_size = len(df)
        print(f"Yeni veri boyutu: {max_data_size}")
        # Base size'ları tekrar filtrele
        valid_base_sizes = [bs for bs in base_sizes if max_data_size >= bs * 2]
    
    # 3. Algoritma listesi
    algorithms = ['malatya', 'kmeans', 'knn', 'rf', 'nb']
    algorithm_names = {
        'malatya': 'Malatya Centrality',
        'kmeans': 'K-Means',
        'knn': 'KNN',
        'rf': 'Random Forest',
        'nb': 'Naive Bayes'
    }
    
    print(f"\nKullanılacak algoritmalar: {[algorithm_names[alg] for alg in algorithms]}")
    
    # 4. Deneyleri başlat
    print("\n[2/5] Deneyler başlatılıyor...")
    print("Bu işlem biraz zaman alabilir...")
    
    all_results = run_all_experiments(df, valid_base_sizes, algorithms)
    
    print(f"\nToplam {len(all_results)} deney tamamlandı.")
    
    # 5. Karşılaştırma tabloları oluştur ve kaydet
    print("\n[3/5] Karşılaştırma tabloları oluşturuluyor...")
    
    # Proje kök dizinini bul
    project_root = os.path.dirname(os.path.dirname(__file__))
    tables_dir = os.path.join(project_root, 'results', 'tables')
    plots_dir = os.path.join(project_root, 'results', 'plots')
    
    os.makedirs(tables_dir, exist_ok=True)
    
    for base_size in valid_base_sizes:
        try:
            table = create_comparison_table(all_results, base_size)
            
            # CSV olarak kaydet
            filename = os.path.join(tables_dir, f'comparison_base_{base_size}.csv')
            table.to_csv(filename, index=False)
            print(f"Tablo kaydedildi: {filename}")
            
            # Ekrana yazdır
            print(f"\n{'='*70}")
            print(f"Base Size: {base_size} - Karşılaştırma Tablosu")
            print(f"{'='*70}")
            print(table.to_string(index=False))
            print()
            
        except Exception as e:
            print(f"Hata (base={base_size}): {e}")
            continue
    
    # 6. Grafikler oluştur
    print("\n[4/5] Grafikler oluşturuluyor...")
    create_all_plots(all_results, valid_base_sizes, save_path=plots_dir)
    
    # 7. Özet istatistikler
    print("\n[5/5] Özet istatistikler hesaplanıyor...")
    
    print("\n" + "="*70)
    print("DENEY ÖZETİ")
    print("="*70)
    
    print(f"\nVeri Boyutu: {max_data_size} satır")
    print(f"Kullanılan Base Boyutları: {valid_base_sizes}")
    print(f"Kullanılan Algoritmalar: {len(algorithms)}")
    print(f"Toplam Deney Sayısı: {len(all_results)}")
    
    # Her base size için deney sayısı
    print("\nBase Size Başına Deney Sayıları:")
    for base_size in valid_base_sizes:
        count = sum(1 for key in all_results.keys() if key[0] == base_size)
        print(f"  Base {base_size}: {count} deney")
    
    # Sonuç dosyaları
    print("\nOluşturulan Dosyalar:")
    print(f"  - Tablolar: {tables_dir}/*.csv")
    print(f"  - Grafikler: {plots_dir}/*.png")
    
    print("\n" + "="*70)
    print("DENEYLER TAMAMLANDI!")
    print("="*70)


if __name__ == "__main__":
    main()

