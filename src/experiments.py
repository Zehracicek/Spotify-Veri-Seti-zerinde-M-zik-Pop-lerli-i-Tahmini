"""
Base Node deneyleri için deney mantığı
"""
import numpy as np
import pandas as pd
from .algorithms import (
    malatya_centrality,
    kmeans_score,
    knn_score,
    random_forest_score,
    naive_bayes_score
)


def run_experiment(df, base_size, algorithm, experiment_id=0):
    """
    Base Node deneyi çalıştırır
    
    Args:
        df: Veri DataFrame'i
        base_size: Base düğüm sayısı
        algorithm: Algoritma adı ('malatya', 'kmeans', 'knn', 'rf', 'nb')
        experiment_id: Deney ID'si (base_size kadar deney yapılacak)
        
    Returns:
        Sıralı sonuç listesi (node, value) ve top sonuçlar
    """
    # Veri boyutunu kontrol et
    if len(df) < base_size * 2:
        print(f"Uyarı: Veri boyutu ({len(df)}) yeterli değil. Base size: {base_size}")
        return []
    
    # Base düğümleri seç (her deney için farklı base seti)
    np.random.seed(42 + experiment_id)
    all_indices = np.arange(len(df))
    base_nodes = np.random.choice(all_indices, size=base_size, replace=False).tolist()
    
    # Base düğümler dışındaki düğümler
    remaining_nodes = [idx for idx in all_indices if idx not in base_nodes]
    
    # Algoritma fonksiyonunu seç
    algorithm_funcs = {
        'malatya': malatya_centrality,
        'kmeans': kmeans_score,
        'knn': knn_score,
        'rf': random_forest_score,
        'nb': naive_bayes_score
    }
    
    if algorithm not in algorithm_funcs:
        raise ValueError(f"Bilinmeyen algoritma: {algorithm}")
    
    algo_func = algorithm_funcs[algorithm]
    
    # Sonuç listesi
    results = []
    
    # Base sabit kalırken, her bir kalan düğümü test et
    # Base size kadar düğüm test edilecek
    test_nodes = remaining_nodes[:base_size] if len(remaining_nodes) >= base_size else remaining_nodes
    
    for new_node in test_nodes:
        try:
            # Algoritma skorunu hesapla
            if algorithm == 'kmeans':
                score = algo_func(df, base_nodes, new_node)
            elif algorithm == 'knn':
                score = algo_func(df, base_nodes, new_node)
            else:
                score = algo_func(df, base_nodes, new_node)
            
            # (node, value) olarak ekle
            results.append((int(new_node), float(score)))
        except Exception as e:
            print(f"Hata (node {new_node}): {e}")
            continue
    
    # Listeyi skora göre sırala (yüksekten düşüğe)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Top sonuçları seç
    if base_size >= 500:
        top_results = results[:20]
    else:
        top_results = results[:10]
    
    return top_results


def run_all_experiments(df, base_sizes, algorithms):
    """
    Tüm base boyutları ve algoritmalar için deneyleri çalıştırır
    
    Args:
        df: Veri DataFrame'i
        base_sizes: Base boyut listesi [10, 20, 50, ...]
        algorithms: Algoritma listesi ['malatya', 'kmeans', ...]
        
    Returns:
        Sonuçlar dictionary: {(base_size, algorithm, exp_id): results}
    """
    all_results = {}
    
    for base_size in base_sizes:
        print(f"\n{'='*60}")
        print(f"Base Size: {base_size}")
        print(f"{'='*60}")
        
        # Veri boyutunu kontrol et
        if len(df) < base_size * 2:
            print(f"Veri boyutu yeterli değil. Base size {base_size} atlanıyor.")
            continue
        
        # Her base size için base_size kadar deney yap
        for exp_id in range(base_size):
            if exp_id % max(1, base_size // 10) == 0:
                print(f"  Deney {exp_id + 1}/{base_size}...")
            
            for algorithm in algorithms:
                try:
                    results = run_experiment(df, base_size, algorithm, experiment_id=exp_id)
                    key = (base_size, algorithm, exp_id)
                    all_results[key] = results
                except Exception as e:
                    print(f"  Hata (base={base_size}, alg={algorithm}, exp={exp_id}): {e}")
                    continue
    
    return all_results


def aggregate_results(all_results, base_size, algorithm):
    """
    Belirli bir base size ve algoritma için tüm deney sonuçlarını birleştirir
    
    Args:
        all_results: Tüm deney sonuçları dictionary
        base_size: Base boyut
        algorithm: Algoritma adı
        
    Returns:
        Birleştirilmiş ve sıralı sonuç listesi
    """
    # İlgili tüm deney sonuçlarını topla
    aggregated = []
    
    for key, results in all_results.items():
        bs, alg, exp_id = key
        if bs == base_size and alg == algorithm:
            aggregated.extend(results)
    
    # Düğüm bazında ortalama skor hesapla (aynı düğüm birden fazla deneyde görülebilir)
    node_scores = {}
    for node, score in aggregated:
        if node not in node_scores:
            node_scores[node] = []
        node_scores[node].append(score)
    
    # Ortalama skorları hesapla
    averaged_results = [(node, np.mean(scores)) for node, scores in node_scores.items()]
    
    # Sırala
    averaged_results.sort(key=lambda x: x[1], reverse=True)
    
    return averaged_results


def create_comparison_table(all_results, base_size):
    """
    Belirli bir base size için tüm algoritmaların karşılaştırma tablosunu oluşturur
    
    Args:
        all_results: Tüm deney sonuçları dictionary
        base_size: Base boyut
        
    Returns:
        Karşılaştırma DataFrame'i
    """
    algorithms = ['malatya', 'kmeans', 'knn', 'rf', 'nb']
    
    # Her algoritma için sonuçları birleştir
    algo_results = {}
    for alg in algorithms:
        aggregated = aggregate_results(all_results, base_size, alg)
        algo_results[alg] = aggregated
    
    # Maksimum rank sayısını belirle
    max_rank = max(len(results) for results in algo_results.values())
    
    # Top sonuç sayısını belirle
    top_n = 20 if base_size >= 500 else 10
    max_rank = min(max_rank, top_n)
    
    # Tablo oluştur
    table_data = {'Rank': range(1, max_rank + 1)}
    
    for alg in algorithms:
        results = algo_results[alg]
        # Rank için node ID'lerini al
        node_ids = [str(node) for node, _ in results[:max_rank]]
        # Eksik rank'ler için boş değer ekle
        while len(node_ids) < max_rank:
            node_ids.append('')
        table_data[alg.capitalize()] = node_ids
    
    df_table = pd.DataFrame(table_data)
    
    return df_table

