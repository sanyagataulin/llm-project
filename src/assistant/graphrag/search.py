"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

_LOCAL_SYSTEM = (
    "Ты ассистент сервисной службы «ПолярТех». Отвечай только по фактам из списка ниже — "
    "каждая строка это связь графа знаний вида «сущность -ОТНОШЕНИЕ-> сущность». "
    "Коды деталей, моделей, ошибок и регламентов пиши точно как в фактах. "
    "Если деталь заменена (REPLACED_BY), называй последнюю деталь цепочки замен. "
    "Последней строкой напиши «Ответ: …» — только суть, одной фразой, например «Ответ: SP-104». "
    "Отказывайся, только если в фактах действительно нет ответа: "
    "тогда единственная строка «в базе знаний не найдено»."
)

_MAP_SYSTEM = (
    "Ты отвечаешь на вопрос по сводке одной части графа знаний сервисной службы «ПолярТех». "
    "Верни JSON: поле answer — что из сводки относится к вопросу, "
    "поле score — целое число от 0 до 10, насколько сводка полезна для ответа. "
    "Ставь score 0, если сводка не помогает ответить или она про другие сущности и линейки, "
    "чем в вопросе."
)

_REDUCE_SYSTEM = (
    "Ты ассистент сервисной службы «ПолярТех». Объедини частичные ответы по разным частям "
    "базы знаний в один точный ответ на вопрос. Перед каждым частичным ответом в скобках "
    "перечислены сущности его части графа — используй только те части, чьи сущности относятся "
    "к вопросу (например, к линейке из вопроса), остальные игнорируй. "
    "Коды и названия пиши точно как в частичных ответах, ничего не придумывай. "
    "Последней строкой напиши «Ответ: …» — только суть, одной фразой. "
    "Если ответа нет, напиши единственную строку: «в базе знаний не найдено»."
)

_MAX_FACTS_CHARS = 8000

_RECHECK = (
    "Проверь список фактов ещё раз: если в нём есть связи, отвечающие на вопрос, "
    "дай последней строкой «Ответ: …». Если ответа действительно нет, повтори "
    "«в базе знаний не найдено»."
)


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    graph = kb.graph
    anchors = _anchors(question, kb)
    if any(_node_id(code_type(code), code) not in graph.nodes for code in find_codes(question)):
        return _refused()
    if not anchors:
        return _refused()

    if len(anchors) >= 2:
        edges = _path_edges(graph, anchors, cutoff=3)
    else:
        edges = _neighbourhood_edges(graph, anchors[0], depth=2)
    nodes = {edge[0] for edge in edges} | {edge[1] for edge in edges}
    edges += _replacement_edges(graph, nodes)
    nodes = {edge[0] for edge in edges} | {edge[1] for edge in edges}
    edges += _supplier_edges(graph, nodes)
    edges = _unique_edges(edges)
    if not edges:
        return _refused()

    facts = _format_facts(graph, edges)
    prompt = "Факты:\n" + "\n".join(f"{i}. {fact}" for i, fact in enumerate(facts, start=1))
    messages = [
        {"role": "system", "content": _LOCAL_SYSTEM},
        {"role": "user", "content": f"{prompt}\n\nВопрос: {question}"},
    ]
    reply = llm.chat(messages)
    if _is_refusal(reply.content):
        # Факты были, но модель отказалась — просим перепроверить список.
        reply = llm.chat(
            [*messages, reply.as_message(), {"role": "user", "content": _RECHECK}]
        )
    sources = sorted({chunk.split("#")[0] for edge in edges for chunk in edge[2]["sources"]})
    return Answer(reply.content, sources=sources, facts=facts, refused=_is_refusal(reply.content))


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    partials: list[str] = []
    sources: set[str] = set()
    for community in kb.communities:
        if not community.summary:
            continue
        members = ", ".join(sorted(kb.graph.nodes[node]["name"] for node in community.nodes))
        reply = llm.chat(
            [
                {"role": "system", "content": _MAP_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Вопрос: {question}\n\nСущности сообщества: {members}\n\n"
                        f"Сводка:\n{community.summary}"
                    ),
                },
            ],
            json_schema=MAP_SCHEMA,
        )
        try:
            data = json.loads(reply.content)
            answer, score = data["answer"], data["score"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if not isinstance(answer, str) or not isinstance(score, int) or not 0 <= score <= 10:
            continue
        if score > 0 and answer.strip():
            partials.append(f"Сущности части: {members}\n{answer.strip()}")
            for node in community.nodes:
                sources.update(chunk.split("#")[0] for chunk in kb.graph.nodes[node].get("sources", []))
    if not partials:
        return _refused()

    prompt = "\n\n".join(f"Частичный ответ {i}:\n{part}" for i, part in enumerate(partials, start=1))
    reply = llm.chat(
        [
            {"role": "system", "content": _REDUCE_SYSTEM},
            {"role": "user", "content": f"{prompt}\n\nВопрос: {question}"},
        ]
    )
    return Answer(reply.content, sources=sorted(sources), refused=_is_refusal(reply.content))


def _refused() -> Answer:
    return Answer("В базе знаний не найдено.", refused=True)


def _is_refusal(text: str) -> bool:
    """Отказ — когда фразой «не найдено» является итоговый ответ, а не слова в середине текста.

    Смотрим на последнюю строку «Ответ: …»; если её нет — на последнюю непустую строку.
    """
    lines = [line.strip() for line in text.lower().splitlines() if line.strip()]
    if not lines:
        return True
    for line in reversed(lines):
        if line.startswith("ответ"):
            return "не найдено" in line
    return "не найдено" in lines[-1]


def _node_id(entity_type: str, name: str) -> str:
    return f"{entity_type}:{name}"


def _anchors(question: str, kb: KnowledgeBase) -> list[str]:
    found: list[str] = []
    for code in find_codes(question):
        node = _node_id(code_type(code), code)
        if node in kb.graph.nodes and node not in found:
            found.append(node)
    for entity_type, name in find_names(question, kb.vocabulary):
        node = _node_id(entity_type, name)
        if node in kb.graph.nodes and node not in found:
            found.append(node)
    return found


def _edge(graph: nx.DiGraph, source: str, target: str) -> list[tuple]:
    """Направленные рёбра графа между двумя узлами в любом направлении."""
    return [
        (u, v, dict(data))
        for u, v in ((source, target), (target, source))
        if graph.has_edge(u, v)
        for data in [graph.edges[u, v]]
    ]


def _path_edges(graph: nx.DiGraph, anchors: list[str], cutoff: int) -> list[tuple]:
    undirected = graph.to_undirected()
    edges: list[tuple] = []
    for i, first in enumerate(anchors):
        for second in anchors[i + 1 :]:
            try:
                paths = nx.all_simple_paths(undirected, first, second, cutoff=cutoff)
            except nx.NodeNotFound:
                continue
            for path in paths:
                for u, v in zip(path, path[1:], strict=False):
                    edges.extend(_edge(graph, u, v))
    return edges


def _neighbourhood_edges(graph: nx.DiGraph, anchor: str, depth: int) -> list[tuple]:
    undirected = graph.to_undirected()
    ego = nx.ego_graph(undirected, anchor, radius=depth)
    edges: list[tuple] = []
    for u, v in ego.edges:
        edges.extend(_edge(graph, u, v))
    return edges


def _replacement_edges(graph: nx.DiGraph, nodes: set[str]) -> list[tuple]:
    edges: list[tuple] = []
    for node in sorted(nodes):
        current = node
        seen = {node}
        while True:
            next_parts = [
                t for _, t, e in graph.out_edges(current, data=True) if e["type"] == "REPLACED_BY"
            ]
            if not next_parts or next_parts[0] in seen:
                break
            target = next_parts[0]
            edges.extend(_edge(graph, current, target))
            seen.add(target)
            current = target
    return edges


def _supplier_edges(graph: nx.DiGraph, nodes: set[str]) -> list[tuple]:
    edges: list[tuple] = []
    for node in sorted(nodes):
        if not node.startswith("Part:"):
            continue
        for _, target, edge in graph.out_edges(node, data=True):
            if edge["type"] == "SUPPLIED_BY":
                edges.append((node, target, dict(edge)))
    return edges


def _unique_edges(edges: list[tuple]) -> list[tuple]:
    seen: set[tuple] = set()
    unique: list[tuple] = []
    for edge in edges:
        key = (edge[0], edge[2].get("type"), edge[1])
        if key not in seen:
            seen.add(key)
            unique.append(edge)
    return unique


def _format_facts(graph: nx.DiGraph, edges: list[tuple]) -> list[str]:
    facts: list[str] = []
    total = 0
    for source, target, edge in edges:
        line = f"{graph.nodes[source]['name']} -{edge['type']}-> {graph.nodes[target]['name']}"
        if total + len(line) + 1 > _MAX_FACTS_CHARS:
            break
        facts.append(line)
        total += len(line) + 1
    return facts
