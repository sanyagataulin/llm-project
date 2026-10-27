"""Сообщества графа и их сводки. Требование FR-3."""

import networkx as nx
from networkx.algorithms.community import louvain_communities

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community

_SEED = 7

_SUMMARY_SYSTEM = (
    "Ты пишешь сводку сообщества графа знаний сервисной службы «ПолярТех». "
    "Опиши до 150 слов: какие сущности входят в сообщество, как они связаны и что встречается чаще всего. "
    "Используй только факты и числа из запроса, ничего не придумывай."
)


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)
    for source, target, edge in graph.edges(data=True):
        weight = edge.get("weight", 1)
        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)

    groups = louvain_communities(undirected, weight="weight", seed=_SEED)
    groups = sorted(groups, key=lambda group: min(group))
    return [Community(index, sorted(nodes)) for index, nodes in enumerate(groups)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    subgraph = graph.subgraph(community.nodes)
    lines = [f"Сообщество из {len(community.nodes)} сущностей."]

    by_type: dict[str, list[str]] = {}
    for node in community.nodes:
        by_type.setdefault(graph.nodes[node]["type"], []).append(graph.nodes[node]["name"])
    for entity_type, names in sorted(by_type.items()):
        lines.append(f"{entity_type} ({len(names)}): {', '.join(sorted(names))}")

    failed = [(u, v, e) for u, v, e in subgraph.edges(data=True) if e["type"] == "FAILED"]
    if failed:
        lines.append("Отказы (число отчётов в скобках):")
        for source, target, edge in sorted(failed, key=lambda item: (-item[2]["weight"], item[0])):
            lines.append(
                f"- {graph.nodes[source]['name']} -FAILED({edge['weight']})-> {graph.nodes[target]['name']}"
            )

    replacements = [(u, v) for u, v, e in subgraph.edges(data=True) if e["type"] == "REPLACED_BY"]
    for source, target in sorted(replacements):
        lines.append(f"- {graph.nodes[source]['name']} заменена на {graph.nodes[target]['name']}")

    reply = llm.chat(
        [
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {"role": "user", "content": "\n".join(lines)},
        ],
        max_tokens=400,
    )
    return " ".join(reply.content.split()[:150])
