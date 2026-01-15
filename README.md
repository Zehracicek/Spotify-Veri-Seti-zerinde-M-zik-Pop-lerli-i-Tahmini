# Spotify Veri Madenciliği Projesi

Bu proje, Spotify veri seti üzerinde farklı veri madenciliği algoritmaları ile deneyler yapar.

## Proje Yapısı

```
vm_final/
├── data/
│   └── spotify_data.csv
├── src/
│   ├── __init__.py
│   ├── loader.py          # Veri yükleme ve temizleme
│   ├── algorithms.py      # Algoritma fonksiyonları
│   ├── experiments.py     # Deney mantığı
│   ├── plotting.py       # Grafik oluşturma
│   └── main.py           # Ana çalıştırma dosyası
├── results/
│   ├── tables/            # CSV tablo sonuçları
│   └── plots/             # PNG grafik sonuçları
├── requirements.txt
└── README.md
```

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

## Kullanım

Projeyi çalıştırmak için:

```bash
cd src
python main.py
```

veya

```bash
python src/main.py
```

## Algoritmalar

Proje aşağıdaki 5 algoritmayı kullanır:

1. **Malatya Centrality**: Düğümler arası mesafe tabanlı merkezilik skoru
2. **K-Means**: Kümeleme tabanlı skor
3. **KNN**: K-En Yakın Komşu skoru
4. **Random Forest**: Rastgele orman regresyon skoru
5. **Naive Bayes**: Naive Bayes sınıflandırma skoru

## Base Node Deneyi

Her veri boyutu (10, 20, 50, 100, 200, 500, 1000, 2000, 5000) için:

- Base size kadar sabit düğüm seçilir
- Base sabit kalırken, veri setinden 1 düğüm eklenir
- Algoritma skoru hesaplanır
- Bu işlem base size kadar tekrarlanır
- Sonuçlar sıralanır ve top-10 veya top-20 seçilir

## Çıktılar

Proje çalıştırıldığında:

- `results/tables/`: Her base size için karşılaştırma tabloları (CSV)
- `results/plots/`: Algoritma karşılaştırma grafikleri (PNG)
- Konsol çıktısı: Deney özeti ve tablolar

## Notlar

- Veri seti çok büyükse (>10000 satır), performans için alt küme alınır
- Her base size için base_size kadar deney yapılır
- Base size >= 500 ise top-20, değilse top-10 sonuç alınır

