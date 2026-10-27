"""Собственные граничные проверки модулей graphrag. Запускаются явным путём."""

import networkx as nx

from assistant.graphrag.communities import detect_communities
from assistant.graphrag.graph_builder import build_graph
from assistant.graphrag.search import global_search, local_search
from assistant.infra.schemas import Community, Entity, ExtractionResult, KnowledgeBase, Relation
from assistant.infra.testing import FakeLLM


def _result(chunk_id: str, *relations: Relation) -> ExtractionResult:
    entities = list({e.node_id: e for r in relations for e in (r.subject, r.object)}.values())
    return ExtractionResult(chunk_id, chunk_id.split("#")[0], entities, relations)


def test_build_graph_empty_input():
    graph = build_graph([])
    assert graph.number_of_nodes() == 0 and graph.graph["conflicts"] == 0


def test_build_graph_ignores_duplicate_chunk_in_weight():
    part = Entity("Part", "P-1042")
    comp = Entity("Component", "пусковое реле")
    relation = Relation(part, "IS_A", comp)
    graph = build_graph([_result("a.md#0", relation), _result("a.md#0", relation)])
    assert graph.edges["Part:P-1042", "Component:пусковое реле"]["weight"] == 1


def test_detect_communities_empty_and_isolated():
    assert detect_communities(nx.DiGraph()) == []
    graph = nx.DiGraph()
    graph.add_node("Part:P-1", type="Part", name="P-1", sources=[])
    communities = detect_communities(graph)
    assert len(communities) == 1 and communities[0].nodes == ["Part:P-1"]


def _relay_kb() -> KnowledgeBase:
    graph = nx.DiGraph(conflicts=0)
    edges = [
        ("Model:AR-210", "HAS_PART", "Part:P-1042", 1),
        ("Part:P-1042", "IS_A", "Component:пусковое реле", 1),
        ("ErrorCode:E-17", "INDICATES", "Component:пусковое реле", 2),
        ("Part:P-1042", "REPLACED_BY", "Part:P-1042B", 1),
        ("Part:P-1042B", "REPLACED_BY", "Part:P-1042C", 1),
        ("Part:P-1042C", "SUPPLIED_BY", "Supplier:ХолодКомплект", 1),
    ]
    for s, kind, t, weight in edges:
        for node in (s, t):
            if node not in graph:
                ntype, name = node.split(":", 1)
                graph.add_node(node, type=ntype, name=name, sources=["d.md#0"])
        graph.add_edge(s, t, type=kind, weight=weight, sources=["d.md#0"])
    return KnowledgeBase(graph, [], {"Component": ["пусковое реле"]})


def test_local_search_returns_last_part_of_replacement_chain():
    kb = _relay_kb()
    llm = FakeLLM(["Закажите P-1042C."])
    answer = local_search("Чем заменить P-1042?", kb, llm)
    prompt = " ".join(str(m["content"]) for m in llm.requests[0]["messages"])
    assert "P-1042C" in prompt and "ХолодКомплект" in prompt and answer.text == "Закажите P-1042C."


def test_local_search_replacement_cycle_does_not_hang():
    kb = _relay_kb()
    kb.graph.add_edge("Part:P-1042C", "Part:P-1042", type="REPLACED_BY", weight=1, sources=["d2.md#0"])
    llm = FakeLLM(["Ответ."])
    assert not local_search("Чем заменить P-1042?", kb, llm).refused


def test_global_search_skips_communities_without_summary():
    kb = _relay_kb()
    kb.communities = [Community(0, sorted(kb.graph.nodes), ""), Community(1, [], "Нет данных.")]
    llm = FakeLLM(['{"answer": "нет", "score": 0}'])
    answer = global_search("Что часто ломается?", kb, llm)
    assert answer.refused and len(llm.requests) == 1
