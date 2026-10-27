"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable
from typing import Any

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.grounding import code_type, normalize_code
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]

_PART_PROPS = {"part_id": {"type": "string", "description": "номер детали, например P-1042 или P-1042B"}}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _tool(
        "search_knowledge",
        "Ищет ответ на вопрос о конкретных моделях, ошибках и деталях по графу знаний из документов.",
        {"question": {"type": "string", "description": "вопрос о документах на русском языке"}},
        ["question"],
    ),
    _tool(
        "summarize_topic",
        "Отвечает на обзорный вопрос по сводкам графа знаний: статистика отказов, снятые детали, общие темы.",
        {"question": {"type": "string", "description": "обзорный вопрос на русском языке"}},
        ["question"],
    ),
    _tool(
        "check_stock",
        "Проверяет остаток детали на складе в учётной системе ERP.",
        {
            **_PART_PROPS,
            "warehouse": {"type": "string", "description": "склад", "enum": WAREHOUSES},
        },
        ["part_id", "warehouse"],
    ),
    _tool(
        "get_price",
        "Возвращает цену детали в рублях и признак снятия с производства из учётной системы ERP.",
        _PART_PROPS,
        ["part_id"],
    ),
    _tool(
        "create_service_order",
        "Создаёт заказ детали на складе. Требует подтверждения диспетчером.",
        {
            **_PART_PROPS,
            "warehouse": {"type": "string", "description": "склад", "enum": WAREHOUSES},
            "qty": {"type": "integer", "description": "количество, больше нуля"},
            "model": {"type": "string", "description": "код модели оборудования, например AR-210"},
        },
        ["part_id", "warehouse", "qty", "model"],
    ),
]


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    warehouses = _warehouses(erp)

    def _part(raw: Any) -> str | None:
        if not isinstance(raw, str):
            return None
        code = normalize_code(raw)
        return code if code and code_type(code) == "Part" else None

    def _model(raw: Any) -> str | None:
        if not isinstance(raw, str):
            return None
        code = normalize_code(raw)
        return code if code and code_type(code) == "Model" else None

    def _warehouse(raw: Any) -> str | None:
        return raw if raw in warehouses else None

    def _call(method: Callable[..., dict], *args: Any, **kwargs: Any) -> dict:
        try:
            return method(*args, **kwargs)
        except (ErpNotFound, ErpRejected) as exc:
            return {"error": str(exc)}

    def search_knowledge(question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "параметр question должен быть непустой строкой"}
        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def summarize_topic(question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "параметр question должен быть непустой строкой"}
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def check_stock(part_id: str, warehouse: str) -> dict:
        part, place = _part(part_id), _warehouse(warehouse)
        if part is None:
            return {"error": f"недопустимый номер детали: {part_id!r}"}
        if place is None:
            return {"error": f"неизвестный склад: {warehouse!r}; допустимые: {', '.join(warehouses)}"}
        return _call(erp.get_stock, part, place)

    def get_price(part_id: str) -> dict:
        part = _part(part_id)
        if part is None:
            return {"error": f"недопустимый номер детали: {part_id!r}"}
        return _call(erp.get_price, part)

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        part, place, equipment = _part(part_id), _warehouse(warehouse), _model(model)
        if part is None:
            return {"error": f"недопустимый номер детали: {part_id!r}"}
        if place is None:
            return {"error": f"неизвестный склад: {warehouse!r}; допустимые: {', '.join(warehouses)}"}
        if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
            return {"error": f"количество должно быть целым числом больше нуля: {qty!r}"}
        if equipment is None:
            return {"error": f"недопустимый код модели: {model!r}"}
        return _call(erp.create_order, part, place, qty, equipment)

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }


def _warehouses(erp: ErpClient) -> list[str]:
    try:
        return list(erp.warehouses())
    except Exception:
        return list(WAREHOUSES)
