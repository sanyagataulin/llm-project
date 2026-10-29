"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, code_type, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import EXTRACTION_SCHEMA, Chunk, Entity, ExtractionResult, Relation

_SYSTEM = """Ты извлекаешь граф знаний из документа сервисной службы «ПолярТех».

Типы сущностей:
- ProductLine – линейка оборудования (Арктика, Бриз, Гольфстрим)
- Model – код модели, например AR-210
- Component – название узла, например пусковое реле
- Part – номер детали, например P-1042 или P-1042B
- ErrorCode – код ошибки, например E-17
- Procedure – номер регламента, например SP-104
- Supplier – название поставщика

Типы связей, направление от subject к object:
- BELONGS_TO: Model -> ProductLine (модель входит в линейку)
- HAS_PART: Model -> Part (в модели установлена деталь)
- IS_A: Part -> Component (деталь является узлом этого вида)
- INDICATES: ErrorCode -> Component (ошибка указывает на отказ узла)
- FIXED_BY: ErrorCode -> Procedure (ошибка устраняется по регламенту)
- REPLACED_BY: Part -> Part (старая деталь заменена на новую; object – это новая деталь)
- SUPPLIED_BY: Part -> Supplier (деталь поставляет поставщик)
- FAILED: Model -> Component (в отчёте о ремонте зафиксирован отказ узла)

Правила:
- Извлекай только факты, которые прямо сказаны в тексте чанка.
- Коды пиши как в тексте; регистр и пробелы не важны.
- Строка «<узел> – деталь <P-код>, поставщик <поставщик>» в разделе о составе узлов даёт
  три связи: модель из заголовка -HAS_PART-> деталь, деталь -IS_A-> узел, деталь -SUPPLIED_BY-> поставщик.
- Соблюдай направление связей из списка выше: IS_A идёт от детали к узлу, HAS_PART от модели к детали.
- Если подходящих фактов нет, верни пустые списки.

Пример.
Текст: «Деталь P-1042 (пусковое реле) снята с производства. Вместо неё устанавливается деталь P-1042B.
Поставщик детали P-1042B – «ХолодКомплект».»
Ответ: {"entities": [{"type": "Part", "name": "P-1042"}, {"type": "Part", "name": "P-1042B"},
{"type": "Component", "name": "пусковое реле"}, {"type": "Supplier", "name": "ХолодКомплект"}],
"relations": [{"subject_type": "Part", "subject": "P-1042", "type": "REPLACED_BY",
"object_type": "Part", "object": "P-1042B"},
{"subject_type": "Part", "subject": "P-1042", "type": "IS_A",
"object_type": "Component", "object": "пусковое реле"},
{"subject_type": "Part", "subject": "P-1042B", "type": "SUPPLIED_BY",
"object_type": "Supplier", "object": "ХолодКомплект"}]}"""

_RETRY = (
    "Ответ не соответствует ожидаемой структуре. Верни один JSON-объект "
    'со списками "entities" и "relations" без пояснений.'
)


class _EntityOut(BaseModel):
    type: str
    name: str


class _RelationOut(BaseModel):
    subject_type: str
    subject: str
    type: str
    object_type: str
    object: str


class _ExtractionOut(BaseModel):
    entities: list[_EntityOut]
    relations: list[_RelationOut]


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    vocab = vocabulary()
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Извлеки сущности и связи из чанка документа {chunk.doc_id}.\n\n{chunk.text}"},
    ]
    for attempt in range(2):
        try:
            reply = llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=4096)
            raw = _ExtractionOut.model_validate(json.loads(reply.content))
        except (json.JSONDecodeError, ValidationError, TypeError):
            if attempt == 0:
                messages = [*messages, {"role": "user", "content": _RETRY}]
            continue
        return _filter(chunk, raw, vocab)
    return ExtractionResult(chunk.chunk_id, chunk.doc_id)


def _filter(chunk: Chunk, raw: _ExtractionOut, vocab: dict[str, list[str]]) -> ExtractionResult:
    """Оставляет только факты, подтверждённые текстом чанка и словарём."""
    codes_in_text = set(find_codes(chunk.text))
    lowered_text = chunk.text.lower()
    entities: dict[str, Entity] = {}
    dropped = 0

    def canon(entity_type: str, raw_name: str) -> Entity | None:
        name = canonical_name(entity_type, raw_name, vocab)
        if name is None:
            return None
        if entity_type in CODE_TYPES:
            if name not in codes_in_text:
                return None
        elif name.lower() not in lowered_text:
            return None
        return Entity(entity_type, name)

    for item in raw.entities:
        entity = canon(item.type, item.name)
        if entity is None:
            dropped += 1
        else:
            entities[entity.node_id] = entity

    relations: list[Relation] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw.relations:
        subject = canon(item.subject_type, item.subject)
        obj = canon(item.object_type, item.object)
        if subject is None or obj is None:
            dropped += 1
            continue
        relation = Relation(subject, item.type, obj)
        if not relation.matches_schema():
            flipped = Relation(obj, item.type, subject)
            if not flipped.matches_schema():
                dropped += 1
                continue
            relation, subject, obj = flipped, obj, subject
        key = (subject.node_id, relation.type, obj.node_id)
        if key in seen:
            continue
        seen.add(key)
        entities.setdefault(subject.node_id, subject)
        entities.setdefault(obj.node_id, obj)
        relations.append(relation)

    _add_manual_parts(chunk, canon, entities, relations, seen)

    return ExtractionResult(chunk.chunk_id, chunk.doc_id, list(entities.values()), relations, dropped)


def _add_manual_parts(
    chunk: Chunk,
    canon: Callable[[str, str], Entity | None],
    entities: dict[str, Entity],
    relations: list[Relation],
    seen: set[tuple[str, str, str]],
) -> None:
    """В руководстве manual_<Model>.md деталь со связью IS_A установлена в эту модель.

    Модель нередко не выписывает HAS_PART, хотя строка «<узел> – деталь <P-код>» означает
    именно установку детали в модель из заголовка. Связь достраивается детерминированно.
    """
    model: Entity | None = None
    for code in find_codes(chunk.doc_id):
        if code_type(code) == "Model":
            model = canon("Model", code)
            break
    if model is None:
        return
    for relation in list(relations):
        if relation.type != "IS_A":
            continue
        part = relation.subject
        key = (model.node_id, "HAS_PART", part.node_id)
        if key in seen:
            continue
        seen.add(key)
        entities.setdefault(model.node_id, model)
        relations.append(Relation(model, "HAS_PART", part))
