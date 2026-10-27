"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import Entity, ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    graph = nx.DiGraph(conflicts=0)

    for result in results:
        for entity in result.entities:
            _add_node(graph, entity, result.chunk_id)
        for relation in result.relations:
            if not relation.matches_schema():
                continue
            for entity in (relation.subject, relation.object):
                _add_node(graph, entity, result.chunk_id)
            source, target = relation.subject.node_id, relation.object.node_id
            if graph.has_edge(source, target):
                edge = graph.edges[source, target]
                edge["sources"] = sorted(set(edge["sources"]) | {result.chunk_id})
                edge["weight"] = len(edge["sources"])
            else:
                graph.add_edge(source, target, type=relation.type, weight=1, sources=[result.chunk_id])

    conflicts = 0
    for source, target, edge in list(graph.edges(data=True)):
        if edge["type"] != "REPLACED_BY" or source >= target:
            continue
        if graph.has_edge(target, source) and graph.edges[target, source]["type"] == "REPLACED_BY":
            conflicts += 1
            backward = graph.edges[target, source]["weight"]
            if edge["weight"] > backward:
                graph.remove_edge(target, source)
            elif edge["weight"] < backward:
                graph.remove_edge(source, target)
            else:
                graph.remove_edge(source, target)
                graph.remove_edge(target, source)
    graph.graph["conflicts"] = conflicts
    return graph


def _add_node(graph: nx.DiGraph, entity: Entity, chunk_id: str) -> None:
    if entity.node_id in graph:
        sources = set(graph.nodes[entity.node_id]["sources"]) | {chunk_id}
        graph.nodes[entity.node_id]["sources"] = sorted(sources)
    else:
        graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=[chunk_id])
