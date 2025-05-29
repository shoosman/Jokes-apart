import random

import spacy
import spacy.cli
from spacy.tokens import Doc, Span, Token

from pymorphy3 import MorphAnalyzer

from typing import List, Tuple, Set, Dict, Optional, Any, Union

Chunks = Dict[Token, Tuple[Tuple[int, int], str]]


def is_russian(text: str) -> bool:
    """
        Проверяет, состоит ли строка только из русских букв, дефисов и пробелов.

        Args:
            text (str): Входная строка.

        Returns:
            bool: True, если строка состоит только из русских букв, дефисов и пробелов, иначе False.
    """

    allowed_symbols = {'-', ' '}

    if not text.strip() or text in allowed_symbols:
        return False

    for char in text:
        if char.isalpha():
            if not ('а' <= char.lower() <= 'я' or char.lower() == 'ё'):
                return False
        elif char not in allowed_symbols:
            return False

    return True


def head_in_named_entity(doc: Doc, span: Span) -> Tuple[Optional[Token], Dict[Token, Token]]:
    """
        Определяет вершину в именованной сущности spaCy.

        Args:
            doc (Doc): Документ spaCy.
            span (Span): Именованная сущность из документа.

        Returns:
            Tuple[Optional[Token], Dict[Token, Token]]: Вершина в именованной сущности, а также словарь, ключами которого являются токены, входящие в именованную сущность, а значениями — их вершины.
    """

    tokens_heads: Dict[Token, Token] = {}
    for token in doc:
        if span.start_char <= token.idx < span.end_char:
            tokens_heads[token] = token.head

    head: Optional[Token] = None
    for token, token_head in tokens_heads.items():
        if token_head not in tokens_heads.keys() or token == token_head:
            head = token
            break

    if head is None and tokens_heads:
        head = random.choice(list(tokens_heads))

    return head, tokens_heads


def normalize_named_entity(doc: Doc, span: Span, head: Token, tokens_heads: Dict[Token, Token]) -> str:
    """
        Приводит именованную сущность spaCy к начальной форме.

        Args:
            doc (Doc): Документ spaCy.
            span (Span): Именованная сущность из документа.
            head (Token): Вершина в именованной сущности.
            tokens_heads (Dict[Token, Token]): Словарь, ключами которого являются токены, входящие в именованную сущность, а значениями — их вершины.

        Returns:
            str: Именованная сущность в начальной форме.
    """

    if head is None:
        return span.text

    head_parse = morph.parse(head.text)
    if not head_parse:
        return span.text
    head_parse = head_parse[0]

    normalized_head = head_parse.inflect({'nomn'})
    if not normalized_head:
        return span.text

    head_tag_parts = str(normalized_head.tag).split(',')
    gender = None
    number = None
    if len(head_tag_parts) > 2:
        head_tag_subparts = head_tag_parts[2].split()
        if len(head_tag_subparts) > 0:
            gender = head_tag_subparts[0]
        if len(head_tag_subparts) > 1:
            number = head_tag_subparts[1]

    normalized_ent = []

    for token, token_head in tokens_heads.items():
        if token == head:
            normalized_ent.append(normalized_head.word)
            continue

        token_parse = morph.parse(token.text)
        if not token_parse:
            return span.text
        token_parse = token_parse[0]

        token_tag = str(token_parse.tag)
        token_pos = token_tag.split(',')[0].split(' ')[0]

        if (token_pos in ('ADJF', 'PRTF') or any(tag in token_tag for tag in ('Surn', 'Patr'))) and token_head == head:
            if gender and number != 'plur':
                normalized_token = token_parse.inflect({gender, 'nomn'})
            else:
                normalized_token = token_parse.inflect({'nomn'})

            if normalized_token:
                normalized_ent.append(normalized_token.word)
                continue

        normalized_ent.append(token_parse.word)

    return ' '.join(normalized_ent).strip()


def extract_named_entities_chunks(doc: Doc) -> Chunks:
    """
        Извлекает из документа spaCy именованные сущности и информацию о них.

        Args:
            doc (Doc): Документ spaCy.

        Returns:
            Chunks: Словарь, ключами которого являются вершины в именованной сущности, а значениями — индексы начального и конечного символов именованной сущности, а также именованные сущности в начальной форме.
    """

    chunks: Chunks = {}

    for ent in doc.ents:
        if is_russian(ent.text):
            head, tokens_heads = head_in_named_entity(doc, ent)
            if head:
                chars = (ent.start_char, ent.end_char)
                chunks[head] = (chars, normalize_named_entity(doc, ent, head, tokens_heads))

    return chunks


def extract_noun_chunks(doc: Doc, chunks: Chunks) -> Chunks:
    """
        Извлекает из документа spaCy имена существительные и информацию о них.

        Args:
            doc (Doc): Документ spaCy.
            сhunks (Chunks): Словарь, ключами которого являются вершины в именованной сущности, а значениями — индексы начального и конечного символов именованной сущности, а также именованные сущности в начальной форме.

        Returns:
            Chunks: Словарь, ключами которого являются токены, а значениями — индексы начального и конечного символов токена, а также концепции в начальной форме.
    """

    for token in doc:
        if is_russian(token.text) and token.pos_ in ('NOUN', 'PROPN') and token not in chunks:
            chars = (token.idx, token.idx + len(token.text))
            chunks[token] = (chars, token.lemma_)

    return chunks


def solve_anaphora(doc: Doc, chunks: Chunks) -> Chunks:
    """
        Разрешает анафору для местоимений в документе spaCy.

        Args:
            doc (Doc): Документ spaCy.
            chunks (Chunks): Словарь, ключами которого являются токены, а значениями — индексы начального и конечного символов токена, а также концепции в начальной форме.

        Returns:
            Chunks: Словарь, ключами которого являются токены, а значениями — индексы начального и конечного символов токена, а также концепции в начальной форме.
    """

    for token in doc:
        if is_russian(token.text) and token.pos_ == 'PRON':
            morph = token.morph.to_dict()
            for key in ('Animacy', 'Case', 'Degree'):
                morph.pop(key, None)

            if not any(pron in token.lemma_ for pron in
                       ('что', 'то', 'либо', 'нибудь')) and not token.text.lower().startswith('вс') and morph.get(
                'Person', 'Third') == 'Third':
                latest_end = 0
                token_value = None

                for chunk_key, chunk_value in chunks.items():
                    if latest_end <= chunk_value[0][0] < token.idx:
                        chunk_morph = chunk_key.morph.to_dict()

                        if all(chunk_morph.get(k, None) == morph[k] for k in morph if k != 'Person'):
                            latest_end = chunk_value[0][1]
                            chars = (token.idx, token.idx + len(token.text))
                            token_value = (chars, chunk_value[1])

                if token_value:
                    chunks[token] = token_value

    return chunks


def check_dep(token: Token, dep: list[str]) -> Optional[Token]:
    """
        По токену возвращает при наличии его дочерний токен с одним из заданных типов зависимости.

        Args:
            token (Token): Токен.
            dep (list[str]): Список типов зависимостей.

        Returns:
            Optional[Token]: Токен с заданным типом зависимости.
    """

    dep_children = [child for child in token.children if child.dep_ in dep]
    if dep_children:
        return dep_children[-1]

    return None


def collect_preds(subj: Token, pred: Token, subjs_preds: Dict[Token, Any], visited: Set[Token] = None) -> None:
    """
        Рекурсивно собирает все сказуемые, относящиеся к одному подлежащему.

        Args:
            subj (Token): Подлежащее.
            pred (Token): Сказуемое.
            subjs_preds (Dict[Token, Any]): Словарь, ключами которого являются сказуемые, а значениями — их подлежащие.
            visited (Set[Token): Список посещённых сказуемых.

        Returns:
            None
    """

    if visited is None:
        visited = set()

    if pred in visited:
        return
    visited.add(pred)

    xcomp_child = check_dep(pred, ['xcomp'])
    if xcomp_child:
        collect_preds(subj, xcomp_child, subjs_preds, visited)
    else:
        subjs_preds[pred] = subj

    conj_child = check_dep(pred, ['conj'])
    if conj_child and not check_dep(conj_child, ['nsubj', 'nsubj:pass']):
        collect_preds(subj, conj_child, subjs_preds, visited)

    advcl_child = check_dep(pred, ['advcl'])
    if advcl_child and not check_dep(advcl_child, ['nsubj', 'nsubj:pass']):
        collect_preds(subj, advcl_child, subjs_preds, visited)


def match_subjs_and_preds(chunks: Chunks) -> Dict[Token, Any]:
    """
        Составляет пары из сказуемых и их подлежащих.

        Args:
            doc (Doc): Документ spaCy.
            chunks (Chunks): Словарь, ключами которого являются токены, а значениями — индексы начального и конечного символов токена, а также концепции в начальной форме.

        Returns:
            Dict[Token, Any]: Словарь, ключами которого являются сказуемые, а значениями — их подлежащие.
    """

    subjs_preds: Union[str, Tuple[str, str, str, str]] = {}

    for chunk_key, chunk_value in chunks.items():
        if chunk_key.dep_ in ('nsubj', 'nsubj:pass'):
            conj_child = check_dep(chunk_key, ['conj'])
            conj = check_dep(conj_child, ['cc']) if conj_child else None

            if conj and conj_child in chunks:
                subj = (chunk_value[1], conj.text, chunks[conj_child][1], 'conj')
            else:
                subj = chunk_value[1]

            collect_preds(subj, chunk_key.head, subjs_preds)

        acl_child = check_dep(chunk_key, ['acl'])
        if acl_child:
            collect_preds(chunk_value[1], acl_child, subjs_preds)

    return subjs_preds


def build_relations(chunks: Chunks, subjs_preds: Dict[Token, Any]) -> List[Tuple]:
    """
        Формирует список отношений между концепциями.

        Args:
            chunks (Chunks): Словарь, ключами которого являются токены, а значениями — индексы начального и конечного символов токена, а также концепции в начальной форме.
            subjs_preds (Dict[Token, Any]): Словарь, ключами которого являются сказуемые, а значениями — их подлежащие.

        Returns:
            List[Tuple]: Список отношений между концепциями.
    """

    relations = []

    for chunk_key, chunk_value in chunks.items():
        if chunk_key.head in subjs_preds and chunk_key.dep_ not in ('nsubj', 'nsubj:pass'):
            conj_child = check_dep(chunk_key, ['conj'])
            conj = check_dep(conj_child, ['cc']) if conj_child else None

            if conj and conj_child in chunks:
                relation_type = 4 if type(subjs_preds[chunk_key.head]) == tuple else 3
                relations.append((relation_type, subjs_preds[chunk_key.head], chunk_key.head.text,
                                  (chunk_value[1], conj.text, chunks[conj_child][1], 'conj'), chunk_key.dep_))

            else:
                relation_type = 2 if type(subjs_preds[chunk_key.head]) == tuple else 1
                relations.append(
                    (relation_type, subjs_preds[chunk_key.head], chunk_key.head.text, chunk_value[1], chunk_key.dep_))

        elif chunk_key.head in chunks and 'nmod' in chunk_key.dep_:
            adp = ''
            case_child = check_dep(chunk_key, ['case'])
            if case_child and case_child.pos_ == 'ADP':
                adp = case_child.text

            conj_child = check_dep(chunk_key, ['conj'])
            conj = check_dep(conj_child, ['cc']) if conj_child else None

            if conj and conj_child in chunks:
                relations.append((3, chunks[chunk_key.head][1], adp,
                                  (chunk_value[1], conj.text, chunks[conj_child][1], 'conj'), chunk_key.dep_))

            else:
                relations.append((1, chunks[chunk_key.head][1], adp, chunk_value[1], chunk_key.dep_))

    return relations


def get_relations(doc: Doc) -> List[Tuple]:
    ent_chunks = extract_named_entities_chunks(doc)
    noun_chunks = extract_noun_chunks(doc, ent_chunks)
    chunks = solve_anaphora(doc, noun_chunks)

    subjs_preds = match_subjs_and_preds(chunks)
    relations = build_relations(chunks, subjs_preds)

    return relations


def main(text: str):
    nlp = spacy.load("ru_core_news_lg")
    doc = nlp(text)

    relations = get_relations(doc)
    print(relations)


if __name__ == '__main__':
    spacy.cli.download("ru_core_news_lg")
    morph = MorphAnalyzer()

    main("Мальчик и девочка пошли в кино.")
