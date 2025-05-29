import os
import re
import pickle
import networkx as nx # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
from graph.higher_dim_graph import Graph, visualize_graph
from graph.edge import Edge
import relations
from relations import get_relations, is_russian
from pymorphy3 import MorphAnalyzer # type: ignore
import spacy # type: ignore

# Инициализация глобальных инструментов
relations.morph = MorphAnalyzer()
morph = relations.morph

nlp = spacy.load('ru_core_news_lg')
nlp.add_pipe('sentencizer')

def safe_str(x):
    return str(x) if pd.notna(x) else ''

def clean_token(s: str) -> str:
    s = re.sub(r'[^А-Яа-яЁё\-]', '', s or '')
    s = re.sub(r'^-+|-+$', '', s)
    s = re.sub(r'-+', '-', s)
    return s

def merge_quoted_sentences(sents):
    merged = []
    buffer = None
    quote_balance = 0

    for sent in sents:
        opens = sent.count('«')
        closes = sent.count('»')

        # Если сейчас не в буфере и это начало цитаты без закрытия
        if buffer is None and opens > closes:
            buffer = sent
            quote_balance = opens - closes

        # Если в буфере (т.е. ждём закрывающую кавычку)
        elif buffer is not None:
            buffer += ' ' + sent
            quote_balance += opens - closes

            # Как только баланс кавычек восстановился — закрыли цитату
            if quote_balance <= 0:
                merged.append(buffer)
                buffer = None
                quote_balance = 0

        # Обычное предложение вне цитат
        else:
            merged.append(sent)

    # На всякий случай, если не закрыли цитату до конца списка
    if buffer is not None:
        merged.append(buffer)

    return merged


def lemmatize_token(s: str) -> str:
    parses = relations.morph.parse(s)
    return parses[0].normal_form if parses else s


def build_and_save_graphs(
    df,
    id_col: str,
    output_dir: str,
    file_prefix: str = None,
    save_png: bool = False,
    png_figsize=(8, 6),
):
    """
    Группируем df по id_col, строим Graph → networkx → сохраняем в GPICKLE
    (и опционально в PNG). Пропускаем те индексы, для которых уже есть
    файл <prefix>_<idx>.gpickle в output_dir.

    Если start_idx > максимального значения в df[id_col], функция завершается сразу.

    Параметры
    ----------
    df : pandas.DataFrame
        Должен содержать столбцы 'pair_type', 'subj', 'pred', 'obj',
        опционально 'union_node', 'relation_label'.
    id_col : str
        Название колонки для группировки (например, 'joke_id').
    output_dir : str
        Папка для сохранения результатов.
    file_prefix : str, optional
        Префикс имени файла. Если None, будет равен id_col.
    save_png : bool
        Сохранять ли ещё и PNG-изображение.
    png_figsize : tuple
        Размер фигуры для визуализации PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = file_prefix or id_col

    # 1) Собираем уже сохранённые номера
    pattern_saved = re.compile(rf'{re.escape(prefix)}_(\d+)\.gpickle$')
    saved = []
    for fname in os.listdir(output_dir):
        m = pattern_saved.match(fname)
        if m:
            saved.append(int(m.group(1)))
    start_idx = max(saved) + 1 if saved else 1
    print(f"Будем обрабатывать начиная с {prefix}_{start_idx}")

    # 2) Ранний выход
    try:
        # из df[id_col] выдергиваем все числа, если они соответствуют шаблону prefix_N
        nums = df[id_col].dropna().map(
            lambda x: int(re.match(rf'{re.escape(prefix)}_(\d+)$', str(x)).group(1))
            if re.match(rf'{re.escape(prefix)}_(\d+)$', str(x)) else None
        ).dropna()
        max_in_df = int(nums.max()) if not nums.empty else None
    except Exception:
        max_in_df = None

    if max_in_df is not None and start_idx > max_in_df:
        print(f"Максимальный номер в данных = {max_in_df}, новых групп нет → выходим.")
        return

    # 3) Обработка групп
    for idx, group in df.groupby(id_col):
        # разбираем числовую часть из idx
        m = re.match(rf'{re.escape(prefix)}_(\d+)$', str(idx))
        if not m:
            # нет совпадения — пропускаем
            continue
        idx_int = int(m.group(1))
        if idx_int < start_idx:
            continue

        # --- строим Graph g по той же логике ---
        g = Graph()
        for _, row in group.iterrows():
            t = int(row.get('pair_type', 0))
            subj           = safe_str(row.get('subj', ''))
            pred           = safe_str(row.get('pred', ''))
            obj            = safe_str(row.get('obj', ''))
            union_node     = safe_str(row.get('union_node', ''))
            relation_label = safe_str(row.get('relation_label', ''))

            g.add_vertex(subj)
            g.add_vertex(obj)

            if t == 1:
                g.add_edge(subj, obj, pred, 0, 0)
            elif t == 2:
                g.add_vertex(union_node)
                g.add_edge(subj, obj, pred, 0, 0)
                e = Edge(subj, obj, pred, 0, 0)
                g.add_union_edge([e], union_node, relation_label, 0)
            elif t == 3:
                g.add_vertex(union_node)
                g.add_edge(subj, obj, pred, 0, 0)
                e = Edge(subj, obj, pred, 0, 0)
                g.add_reversed_union_edge(union_node, [e], relation_label, 0)
            else:
                continue

        # Конвертация в networkx
        G_nx = nx.DiGraph()
        for v in g.vertices:
            G_nx.add_node(v)
        for e in g.edges:
            rel = safe_str(e.meaning)
            G_nx.add_edge(e.agent_1, e.agent_2, predicate=rel)

        # Сохраняем, используя оригинальный idx («joke_2»)
        pkl_name = f"{idx}.gpickle"
        pkl_path = os.path.join(output_dir, pkl_name)
        with open(pkl_path, 'wb') as f:
            pickle.dump(G_nx, f)

        if save_png:
            plt.figure(figsize=png_figsize)
            visualize_graph(g)
            png_name = f"{idx}.png"
            png_path = os.path.join(output_dir, png_name)
            plt.savefig(png_path, bbox_inches='tight', dpi=100)
            plt.close()
            print(f'[{idx}] → GPICKLE: {pkl_path}, PNG: {png_path}')
        else:
            print(f'[{idx}] → GPICKLE: {pkl_path}')



def extract_relations(
    df: pd.DataFrame,
    id_col: str,
    output_raw: str,
    output_clean: str,
    max_unique: int = None,
    stop_on_limit: bool = False
) -> pd.DataFrame:
    """
    Извлекает тройки отношений из текстов в df, сохраняет raw и clean CSV.

    Параметры
    ----------
    df : DataFrame
        Должен содержать столбцы id_col и 'sentence'.
    id_col : str
        Название столбца идентификатора (e.g. 'joke_id' или 'segment_id').
    output_raw : str
    output_clean : str
    max_unique : int, optional
        Максимальное число уникальных id для обработки.
    stop_on_limit : bool
        Если True, прерывать при достижении max_unique.

    Возвращает
    ----------
    cleaned DataFrame
    """
    rows = []
    seen_ids = set()
    stop = False

    for _, row in df.iterrows():
        idx = row[id_col]
        if max_unique is not None and idx not in seen_ids:
            seen_ids.add(idx)
            if stop_on_limit and len(seen_ids) >= max_unique:
                print(f"Достигли лимита в {max_unique} уникальных {id_col} — прерываем.")
                stop = True
        if stop:
            break

        text = str(row['sentence'])
        doc = nlp(text)
        for sent in doc.sents:
            relations_list = get_relations(sent)
            for rel in relations_list:
                t = rel[0]
                if t == 1:
                    subj, pred, obj = rel[1], rel[2], rel[3]
                    union_node = ''
                    rel_label = rel[4]
                elif t == 2:
                    triple = rel[1]
                    subj, pred, obj = triple[:3]
                    rel_label = rel[2]
                    union_node = rel[3]
                elif t == 3:
                    union_node = rel[1]
                    triple = rel[3]
                    subj, pred, obj = triple[:3]
                    rel_label = rel[2]
                else:
                    continue

                # Очистка и лемматизация
                subj = lemmatize_token(clean_token(subj))
                pred = lemmatize_token(clean_token(pred))
                obj = lemmatize_token(clean_token(obj))

                # Фильтрация
                if not (is_russian(subj) and is_russian(obj)):
                    continue
                if (len(subj) < 2 and subj != 'я') or len(obj) < 2:
                    continue
                if subj == obj or subj == pred or obj == pred:
                    # если есть union_node, оставляем
                    if not union_node:
                        continue

                rows.append({
                    id_col: idx,
                    'pair_type': t,
                    'subj': subj,
                    'pred': pred,
                    'obj': obj,
                    'union_node': union_node,
                    'relation_label': rel_label
                })

    out_raw = pd.DataFrame(rows)
    out_raw.to_csv(output_raw, index=False, encoding='utf-8')
    print(f"Извлечено {len(out_raw)} отношений, сохранено в {output_raw}")

    out_clean = out_raw.drop_duplicates(subset=[id_col, 'subj', 'pred', 'obj'])
    out_clean.to_csv(output_clean, index=False, encoding='utf-8')
    print(f"После фильтрации и лемматизации осталось {len(out_clean)} отношений, сохранено в {output_clean}")

    return out_clean