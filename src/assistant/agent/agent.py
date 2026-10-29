"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import _CODE, find_codes, normalize_code
from assistant.infra.llm import LLMClient, ToolCall, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

_SYSTEM = (
    "Ты — ассистент сервисной службы «ПолярТех». Отвечай диспетчеру, используя инструменты.\n"
    "Правила:\n"
    "- Сначала ищи в базе знаний: search_knowledge для конкретных вопросов, summarize_topic для обзорных.\n"
    "- Если в вопросе есть код (E-17, P-1042, AR-210, SP-104) или название узла — "
    "первым делом вызови search_knowledge, всегда.\n"
    "- Если summarize_topic ответил «не найдено», вызови search_knowledge с тем же вопросом.\n"
    "- Остатки и цены известны только в ERP: check_stock и get_price. Никогда не называй их по памяти.\n"
    "- Номер детали бери только из вопроса или результатов инструментов.\n"
    "- Заказ create_service_order вызывай, только когда деталь, склад и количество точно известны.\n"
    "- Не пиши «в базе знаний не найдено», пока не вызван подходящий инструмент.\n"
    "- Если ответа на вопрос нет в результатах инструментов — честно откажись и заверши ответ "
    "последней строкой «Ответ: в базе знаний не найдено». Если сущность нашлась, а для заказа "
    "не хватает количества или склада, назови найденное и уточни параметры — это не отказ."
)

_LIMIT_NOTE = (
    "Лимит вызовов инструментов исчерпан. Сформулируй ответ по уже полученным результатам "
    "и скажи, что работа прервана."
)
_NO_SEARCH_NOTE = (
    "Ты ответил отказом, но в вопросе есть код сущности, а search_knowledge не вызывался. "
    "Вызови search_knowledge с вопросом пользователя и только потом отвечай."
)
_CORRECTION = (
    "В твоём ответе есть факты, которых нет ни в вопросе, ни в результатах инструментов: {facts}. "
    "Исправь ответ: используй только данные из вопроса и результатов инструментов."
)

_NUMBER = r"(?<![\w-])(?:\d{1,3}(?:[   ]\d{3})+|\d+)(?![\w-])"
_WITH_UNIT = re.compile(rf"({_NUMBER})\s*(руб|₽|шт)")


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": question}]
    steps: list[ToolStep] = []
    sources: set[str] = set()
    cache: dict[str, dict] = {}
    evidence_codes = set(find_codes(question))
    evidence_numbers = _numbers(question)
    question_numbers = _numbers(question)

    response = None
    budget = max_steps
    pending: dict | None = None
    nudged = False
    while response is None or response.tool_calls:
        if response is not None:
            for call in response.tool_calls:
                if budget <= 0:
                    break
                budget -= 1
                result, error, pending = _execute(
                    call, tools, evidence_codes, question_numbers, cache, confirm
                )
                step = ToolStep(call.name, call.arguments, result, error)
                steps.append(step)
                if result is not None:
                    sources.update(result.get("sources") or [])
                    evidence_codes.update(find_codes(json.dumps(result, ensure_ascii=False)))
                    evidence_numbers.update(re.findall(r"\d+", json.dumps(result, ensure_ascii=False)))
                    cache[_cache_key(call.name, call.arguments)] = result
                messages.append(tool_message(call, result if result is not None else {"error": error}))
                if pending is not None:
                    return AgentResult(
                        _pending_text(pending),
                        steps,
                        sorted(sources),
                        needs_confirmation=True,
                        pending_action=pending,
                    )
            if budget <= 0:
                messages.append({"role": "user", "content": _LIMIT_NOTE})
                for _ in range(3):
                    response = llm.chat(messages)
                    messages.append(response.as_message())
                    if not response.tool_calls:
                        break
                break
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(response.as_message())
        # Отказ без единого поиска по коду из вопроса — подталкиваем модель к search_knowledge.
        if (
            not response.tool_calls
            and not nudged
            and budget > 0
            and _is_refusal(response.content)
            and find_codes(question)
            and not any(step.tool == "search_knowledge" for step in steps)
        ):
            nudged = True
            messages.append({"role": "user", "content": _NO_SEARCH_NOTE})
            response = None

    answer = response.content or "Работа прервана: исчерпан лимит вызовов инструментов."
    for _ in range(1):
        problems = _answer_problems(answer, evidence_codes, evidence_numbers)
        if not problems:
            break
        messages.append({"role": "user", "content": _CORRECTION.format(facts=", ".join(problems))})
        reply = llm.chat(messages)
        if _answer_problems(reply.content, evidence_codes, evidence_numbers):
            return AgentResult("В базе знаний не найдено.", steps, sorted(sources), refused=True)
        answer = reply.content

    return AgentResult(answer, steps, sorted(sources), refused=_is_refusal(answer))


def _cache_key(name: str, args: dict) -> str:
    return f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"


def _execute(
    call: ToolCall,
    tools: dict[str, Callable[..., dict]],
    evidence_codes: set[str],
    question_numbers: set[str],
    cache: dict[str, dict],
    confirm: bool,
) -> tuple[dict | None, str | None, dict | None]:
    """Один вызов инструмента с проверками. Возвращает (result, error, pending_action)."""
    name, args = call.name, call.arguments
    if name not in tools:
        return None, f"неизвестный инструмент {name!r}", None
    if not isinstance(args, dict) or "_raw" in args:
        return None, "нечитаемые аргументы вызова", None

    part = args.get("part_id")
    if part is not None:
        code = normalize_code(str(part)) or ""
        if code not in evidence_codes:
            return (
                None,
                f"деталь {part!r} не встречается в вопросе и результатах; сначала вызовите search_knowledge",
                None,
            )
        args = {**args, "part_id": code}

    if name == "create_service_order":
        qty = args.get("qty")
        if not isinstance(qty, int) or isinstance(qty, bool) or str(qty) not in question_numbers:
            return None, "количество не названо в вопросе; уточните у пользователя, сколько штук заказать", None
        if not confirm:
            return None, "заказ ожидает подтверждения пользователя", {"tool": name, "args": args}

    key = _cache_key(name, args)
    if key in cache:
        return cache[key], None, None

    last_error = None
    for _attempt in range(3):
        try:
            return tools[name](**args), None, None
        except ErpUnavailable as exc:
            last_error = f"ERP недоступна: {exc}"
        except TypeError as exc:
            return None, f"неверные аргументы: {exc}", None
    return None, last_error, None


def _numbers(text: str) -> set[str]:
    """Числа текста вне кодов вида P-1042B, без разделителей разрядов."""
    masked = list(text)
    for match in _CODE.finditer(text):
        for i in range(*match.span()):
            masked[i] = " "
    return {re.sub(r"\D", "", raw) for raw in re.findall(_NUMBER, "".join(masked))}


def _answer_problems(answer: str, evidence_codes: set[str], evidence_numbers: set[str]) -> list[str]:
    problems = [f"код {code}" for code in find_codes(answer) if code not in evidence_codes]
    for raw, unit in _WITH_UNIT.findall(answer):
        digits = re.sub(r"\D", "", raw)
        if digits not in evidence_numbers:
            problems.append(f"{digits} {unit}")
    return problems


def _pending_text(pending: dict) -> str:
    args = pending["args"]
    return (
        f"Требуется подтверждение: создать заказ на деталь {args.get('part_id')} "
        f"в количестве {args.get('qty')} шт. для {args.get('model')} на складе {args.get('warehouse')}."
    )


def _is_refusal(text: str) -> bool:
    """Отказ — когда фразой «не найдено» является итоговый ответ, а не слова в середине текста."""
    lines = [line.strip() for line in text.lower().splitlines() if line.strip()]
    if not lines:
        return True
    for line in reversed(lines):
        if line.startswith("ответ"):
            return "не найдено" in line
    return "не найдено" in lines[-1]
