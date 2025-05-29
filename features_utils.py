import networkx as nx
import pandas as pd
import glob
import os
import random
import pickle
import numpy as np
import warnings
from collections import Counter
from scipy.stats import entropy

# Константы для выборки
SEED = 42


# Функция расчёта признаков триплетов
def optimized_triplet_features(df, id_column):
    df = df.copy()
    df['subj_len'] = df['subj'].str.len()
    df['pred_len'] = df['pred'].str.len()
    df['obj_len'] = df['obj'].str.len()

    features = df.groupby(id_column).agg(
        avg_subj_len=('subj_len', 'mean'),
        avg_pred_len=('pred_len', 'mean'),
        avg_obj_len=('obj_len', 'mean'),
        unique_subj_count=('subj', pd.Series.nunique),
        unique_pred_count=('pred', pd.Series.nunique),
        unique_obj_count=('obj', pd.Series.nunique)
    ).reset_index()

    return features

def triplet_extra_features(df, id_col):
    rows = []
    for gid, g in df.groupby(id_col):
        n = len(g)
        up = g['pred'].nunique()
        uo = g['obj'].nunique()
        us = g['subj'].nunique()

        # Новые метрики
        subj_obj_ratio   = us / uo         if uo else np.nan
        pred_edge_ratio  = up / n          if n   else np.nan

        pred_ent = entropy(list(Counter(g['pred']).values()), base=2)
        subj_ent = entropy(list(Counter(g['subj']).values()), base=2)
        obj_ent  = entropy(list(Counter(g['obj']).values()),  base=2)

        rows.append({
            id_col:            gid,
            'subj_obj_ratio':  subj_obj_ratio,
            'pred_edge_ratio': pred_edge_ratio,
            'subj_entropy':    subj_ent,
            'pred_entropy':    pred_ent,
            'obj_entropy':     obj_ent
        })
    return pd.DataFrame(rows)

def graph_features(G):
    """
    Вычисление базовых графовых признаков для Directed графа G.
    Возвращает словарь с числами узлов, рёбер, средней степенью, плотностью,
    числом компонент, средним размером компоненты,
    средней длиной кратчайшего пути и диаметром (для связного графа).
    """
    # Число вершин и рёбер
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Средняя степень (in+out degree)
    avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes else 0

    # Плотность для ориентированного графа
    density = nx.density(G)

    # Слабые компоненты связности
    components = list(nx.weakly_connected_components(G))
    num_components = len(components)
    avg_component_size = sum(len(c) for c in components) / num_components if num_components else 0

    # Подготовим метрики пути на неориентированном графе
    avg_shortest_path = None
    diameter = None
    H = G.to_undirected()
    if nx.is_connected(H):
        avg_shortest_path = nx.average_shortest_path_length(H)
        diameter = nx.diameter(H)

    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': avg_degree,
        'density': density,
        'num_components': num_components,
        'avg_component_size': avg_component_size,
        'avg_shortest_path': avg_shortest_path,
        'diameter': diameter
    }


def process_graph_folder(folder_path, label_prefix):
    """
    Обработка всех файлов .gpickle из folder_path без выборки
    и вычисление признаков через graph_features.
    Возвращает DataFrame с признаками и колонкой <label_prefix>_id.
    """
    rows = []
    for filepath in glob.glob(os.path.join(folder_path, '*.gpickle')):
        graph_id = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, 'rb') as f:
            G = pickle.load(f)
        feats = graph_features(G)
        feats[f'{label_prefix}_id'] = graph_id
        rows.append(feats)
    return pd.DataFrame(rows)


def extended_graph_features(G):
    """
    Вычисление расширенных графовых признаков для графа G.
    Включает кластеризацию, транзитивность, треугольники,
    центральности, количество листьев, ассортативность,
    ядра k-core, спектральный радиус и радиус графа.
    """
    n = G.number_of_nodes()
    UG = G.to_undirected()

    avg_clust = nx.average_clustering(UG) if n else 0
    trans = nx.transitivity(UG)
    num_triangles = sum(nx.triangles(UG).values()) / 3

    # Degree centrality
    deg_cent = nx.degree_centrality(G)
    avg_deg_cent = np.mean(list(deg_cent.values())) if deg_cent else 0
    max_deg_cent = max(deg_cent.values(), default=0)

    # Closeness centrality
    clos_cent = nx.closeness_centrality(G)
    avg_clos_cent = np.mean(list(clos_cent.values())) if clos_cent else 0
    max_clos_cent = max(clos_cent.values(), default=0)

    # Betweenness centrality
    betw_cent = nx.betweenness_centrality(G)
    avg_betw = np.mean(list(betw_cent.values())) if betw_cent else 0
    max_betw = max(betw_cent.values(), default=0)

    # Количество листьев
    leaf_cnt = sum(1 for _, d in G.degree() if d == 1)

    # Assortativity
    assort = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            val = nx.degree_assortativity_coefficient(G)
        if np.isfinite(val):
            assort = val
    except:
        pass

    # k-core
    try:
        cores = nx.core_number(UG)
        max_core = max(cores.values(), default=0)
    except:
        max_core = None

    # Spectral radius
    try:
        A = nx.to_numpy_array(G)
        eigs = np.linalg.eigvals(A)
        spectral_rad = max(np.abs(eigs))
    except:
        spectral_rad = None

    # Радиус графа
    try:
        radius = nx.radius(UG) if nx.is_connected(UG) else None
    except:
        radius = None

    return {
        'avg_clustering': avg_clust,
        'transitivity': trans,
        'num_triangles': num_triangles,
        'avg_degree_centrality': avg_deg_cent,
        'max_degree_centrality': max_deg_cent,
        'avg_closeness_centrality': avg_clos_cent,
        'max_closeness_centrality': max_clos_cent,
        'avg_betweenness_centrality': avg_betw,
        'max_betweenness_centrality': max_betw,
        'leaf_node_count': leaf_cnt,
        'assortativity': assort,
        'max_k_core': max_core,
        'spectral_radius': spectral_rad,
        'graph_radius': radius
    }


def compute_extended_features(folder_path, id_col):
    """
    Сборка DataFrame расширенных признаков из файлов .gpickle в папке folder_path.
    Возвращает DataFrame с колонкой id_col.
    """
    records = []
    for path in glob.glob(os.path.join(folder_path, '*.gpickle')):
        graph_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, 'rb') as f:
            G = pickle.load(f)
        feats = extended_graph_features(G)
        feats[id_col] = graph_id
        records.append(feats)
    return pd.DataFrame(records)

def graph_extra_features(G):
    feats = {}
    # приводим к неориентированному для топологических метрик
    UG = G.to_undirected()

    n = G.number_of_nodes()
    # 1) Индекс централизации по степени:
    degs = np.array(list(dict(G.degree()).values()))
    max_deg = degs.max() if n else 0
    mean_deg = degs.mean() if n else 0
    # максимально возможная разница сумм: (n-1) для самого центричного графа
    feats['degree_centralization'] = ((max_deg - degs).sum() 
                                      / ((n-1)*(n-2))) if n>2 else 0

    # 2) Число мостов и артикуляционных точек
    feats['bridge_count'] = sum(1 for _ in nx.bridges(UG))
    feats['articulation_count'] = sum(1 for _ in nx.articulation_points(UG))

    # 3) Размер максимальной клики
    try:
        feats['max_clique_size'] = max((len(c) for c in nx.find_cliques(UG)), default=0)
    except Exception:
        feats['max_clique_size'] = 0

    # 4) Reciprocity (для ориентированных графов)
    #   доля пар ребёр i->j и j->i
    if G.is_directed():
        feats['reciprocity'] = nx.reciprocity(G)
    else:
        feats['reciprocity'] = np.nan

    # возвращаем словарь
    return feats

def append_graph_extra(input_csv, graphs_folder, id_col):
    rows = []
    # 1) собираем доп. признаки из каждого .gpickle
    for path in glob.glob(f'{graphs_folder}/*.gpickle'):
        gid = path.split('/')[-1].replace('.gpickle','')
        with open(path, 'rb') as f:
            G = pickle.load(f)
        ex = graph_extra_features(G)
        ex[id_col] = gid
        rows.append(ex)
    ext_df = pd.DataFrame(rows)
    # 2) объединяем с уже существующим CSV
    base = pd.read_csv(input_csv)
    merged = base.merge(ext_df, on=id_col, how='left')
    merged.to_csv(input_csv, index=False)
