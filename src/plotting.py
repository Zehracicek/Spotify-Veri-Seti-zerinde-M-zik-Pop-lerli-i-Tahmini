"""
Grafik oluşturma modülü
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from .experiments import aggregate_results


def plot_base_comparison(all_results, base_size, save_path='results/plots/'):
    """
    Belirli bir base size için algoritmalar arası karşılaştırma grafiği
    
    Args:
        all_results: Tüm deney sonuçları dictionary
        base_size: Base boyut
        save_path: Grafik kaydetme yolu
    """
    algorithms = ['malatya', 'kmeans', 'knn', 'rf', 'nb']
    algorithm_names = {
        'malatya': 'Malatya Centrality',
        'kmeans': 'K-Means',
        'knn': 'KNN',
        'rf': 'Random Forest',
        'nb': 'Naive Bayes'
    }
    
    # Her algoritma için ortalama skorları hesapla
    algo_avg_scores = {}
    for alg in algorithms:
        aggregated = aggregate_results(all_results, base_size, alg)
        if aggregated:
            scores = [score for _, score in aggregated]
            algo_avg_scores[alg] = np.mean(scores)
        else:
            algo_avg_scores[alg] = 0
    
    # Grafik oluştur
    plt.figure(figsize=(10, 6))
    alg_names = [algorithm_names[alg] for alg in algorithms]
    scores = [algo_avg_scores[alg] for alg in algorithms]
    
    bars = plt.bar(alg_names, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    plt.title(f'Algoritma Karşılaştırması - Base Size: {base_size}', fontsize=14, fontweight='bold')
    plt.xlabel('Algoritma', fontsize=12)
    plt.ylabel('Ortalama Skor', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Değerleri çubukların üzerine yaz
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Kaydet
    os.makedirs(save_path, exist_ok=True)
    filename = f'comparison_base_{base_size}.png'
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Grafik kaydedildi: {filepath}")
    plt.close()


def plot_line_comparison(all_results, base_sizes, save_path='results/plots/'):
    """
    Tüm base boyutları için algoritmaların performans çizgi grafiği
    
    Args:
        all_results: Tüm deney sonuçları dictionary
        base_sizes: Base boyut listesi
        save_path: Grafik kaydetme yolu
    """
    algorithms = ['malatya', 'kmeans', 'knn', 'rf', 'nb']
    algorithm_names = {
        'malatya': 'Malatya Centrality',
        'kmeans': 'K-Means',
        'knn': 'KNN',
        'rf': 'Random Forest',
        'nb': 'Naive Bayes'
    }
    
    # Her base size ve algoritma için ortalama skor hesapla
    plot_data = {alg: [] for alg in algorithms}
    valid_base_sizes = []
    
    for base_size in base_sizes:
        has_data = False
        for alg in algorithms:
            aggregated = aggregate_results(all_results, base_size, alg)
            if aggregated:
                scores = [score for _, score in aggregated]
                avg_score = np.mean(scores)
                plot_data[alg].append(avg_score)
                has_data = True
            else:
                plot_data[alg].append(0)
        
        if has_data:
            valid_base_sizes.append(base_size)
    
    # Grafik oluştur
    plt.figure(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, alg in enumerate(algorithms):
        if valid_base_sizes:
            scores = plot_data[alg][:len(valid_base_sizes)]
            plt.plot(valid_base_sizes, scores, 
                    label=algorithm_names[alg], 
                    marker=markers[idx], 
                    color=colors[idx],
                    linewidth=2,
                    markersize=8)
    
    plt.title('Algoritma Performans Karşılaştırması - Base Size Değişimi', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Base Size', fontsize=12)
    plt.ylabel('Ortalama Skor', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')  # Log scale daha iyi görünüm için
    plt.tight_layout()
    
    # Kaydet
    os.makedirs(save_path, exist_ok=True)
    filename = 'algorithm_comparison_all_bases.png'
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Grafik kaydedildi: {filepath}")
    plt.close()


def plot_algorithm_performance(all_results, algorithm, base_sizes, save_path='results/plots/'):
    """
    Belirli bir algoritma için base size'a göre performans grafiği
    
    Args:
        all_results: Tüm deney sonuçları dictionary
        algorithm: Algoritma adı
        base_sizes: Base boyut listesi
        save_path: Grafik kaydetme yolu
    """
    algorithm_names = {
        'malatya': 'Malatya Centrality',
        'kmeans': 'K-Means',
        'knn': 'KNN',
        'rf': 'Random Forest',
        'nb': 'Naive Bayes'
    }
    
    avg_scores = []
    std_scores = []
    valid_base_sizes = []
    
    for base_size in base_sizes:
        aggregated = aggregate_results(all_results, base_size, algorithm)
        if aggregated:
            scores = [score for _, score in aggregated]
            avg_scores.append(np.mean(scores))
            std_scores.append(np.std(scores))
            valid_base_sizes.append(base_size)
    
    if not valid_base_sizes:
        print(f"Grafik oluşturulamadı: {algorithm} için veri yok")
        return
    
    # Grafik oluştur
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(valid_base_sizes, avg_scores, yerr=std_scores, 
                marker='o', linewidth=2, markersize=8, capsize=5,
                label=algorithm_names.get(algorithm, algorithm))
    
    plt.title(f'{algorithm_names.get(algorithm, algorithm)} Performansı', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Base Size', fontsize=12)
    plt.ylabel('Ortalama Skor ± Standart Sapma', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    
    # Kaydet
    os.makedirs(save_path, exist_ok=True)
    filename = f'{algorithm}_performance.png'
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Grafik kaydedildi: {filepath}")
    plt.close()


def create_all_plots(all_results, base_sizes, save_path='results/plots/'):
    """
    Tüm grafikleri oluşturur
    
    Args:
        all_results: Tüm deney sonuçları dictionary
        base_sizes: Base boyut listesi
        save_path: Grafik kaydetme yolu
    """
    print("\n" + "="*60)
    print("Grafikler oluşturuluyor...")
    print("="*60)
    
    # Her base size için karşılaştırma grafiği
    for base_size in base_sizes:
        plot_base_comparison(all_results, base_size, save_path)
    
    # Tüm base boyutları için çizgi grafiği
    plot_line_comparison(all_results, base_sizes, save_path)
    
    # Her algoritma için performans grafiği
    algorithms = ['malatya', 'kmeans', 'knn', 'rf', 'nb']
    for alg in algorithms:
        plot_algorithm_performance(all_results, alg, base_sizes, save_path)
    
    print("\nTüm grafikler oluşturuldu!")

