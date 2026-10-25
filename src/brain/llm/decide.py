"""Typed decisions: closed questions about a text, answered with a probability per option.

``LLM.decide`` posts to ``/v1/decisions`` (OpenAI Decisions-compatible; oMLX serves it for any
chat model by reading the next-token probabilities of option letters, one token per question).
A runtime without the endpoint gets one ``structured`` call with an enum per question instead:
a label without a probability, marked ``fallback`` (confidence 1.0).

Questions describe the evidence, never a proposed answer: shown "works at: X?" the model just
agrees (docs/decision-models.md).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, create_model

YES, NO = "yes", "no"


class Choice(BaseModel):
    value: str
    description: str = ""


class ChoiceQuestion(BaseModel):
    """One value out of unordered options (give an explicit "other" when none may fit)."""

    type: Literal["choice"] = "choice"
    name: str
    instructions: str
    choices: list[Choice]


class PredicateQuestion(BaseModel):
    """The probability that a statement about the input is true."""

    type: Literal["predicate"] = "predicate"
    name: str
    instructions: str


Question = Annotated[ChoiceQuestion | PredicateQuestion, Field(discriminator="type")]


class Decision(BaseModel):
    """The answer to one question.

    ``probabilities`` maps every option to its probability (a predicate has ``yes``/``no``).
    ``refusal``: the model did not answer (its option letters held under half the next-token
    mass); there is no choice then. ``fallback``: answered by a plain structured call (runtime
    without ``/v1/decisions``), so the probability is 1.0 for the label, not a measurement.
    """

    name: str
    type: Literal["choice", "predicate"]
    choice: str | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0
    refusal: bool = False
    fallback: bool = False

    @property
    def p_yes(self) -> float | None:
        """A predicate's probability of "yes" (None on a refusal)."""
        return None if self.refusal else self.probabilities.get(YES)

    def summary(self) -> str:
        """``kind=project (conf 0.97)`` for notes and logs."""
        if self.refusal:
            return f"{self.name}: no answer (refusal)"
        tag = ", fallback" if self.fallback else ""
        if self.type == "predicate":
            return f"{self.name}: p(yes)={self.p_yes:.3f}{tag}"
        return f"{self.name}={self.choice} (conf {self.confidence:.2f}{tag})"


def request_payload(model: str, text: str, questions: Sequence[Question]) -> dict[str, Any]:
    return {
        "model": model,
        "input": text,
        "questions": [_question_json(q) for q in questions],
    }


def _question_json(q: ChoiceQuestion | PredicateQuestion) -> dict[str, Any]:
    if isinstance(q, PredicateQuestion):
        return {"type": "predicate", "name": q.name, "instructions": q.instructions}
    return {
        "type": "choice",
        "name": q.name,
        "instructions": q.instructions,
        "choices": [
            {"value": c.value, **({"description": c.description} if c.description else {})}
            for c in q.choices
        ],
    }


def parse_answers(
    data: Mapping[str, Any], questions: Sequence[ChoiceQuestion | PredicateQuestion]
) -> list[Decision]:
    """``/v1/decisions`` response → one ``Decision`` per question, in question order (a question
    the server did not answer is a refusal)."""
    by_name = {a.get("name"): a for a in data.get("answers") or []}
    out = []
    for q in questions:
        answer = by_name.get(q.name) or {"type": "refusal"}
        if answer.get("type") == "refusal":
            out.append(Decision(name=q.name, type=q.type, refusal=True))
        elif q.type == "predicate":
            p = float(answer.get("probability", 0.0))
            out.append(
                Decision(
                    name=q.name,
                    type="predicate",
                    choice=YES if p >= 0.5 else NO,
                    probabilities={YES: p, NO: round(1.0 - p, 4)},
                    confidence=abs(2 * p - 1),
                )
            )
        else:
            probs = {str(p["value"]): float(p["probability"]) for p in answer["probabilities"]}
            out.append(
                Decision(
                    name=q.name,
                    type="choice",
                    choice=str(answer["choice"]),
                    probabilities=probs,
                    confidence=float(answer.get("confidence", 0.0)),
                )
            )
    return out


# --- fallback: one structured call ------------------------------------------------------------

FALLBACK_SYSTEM = (
    # decide_fallback v1: used only when the runtime has no /v1/decisions.
    "You answer closed questions about a text. For each question, pick exactly one of its "
    "options. Fields: reasoning (one short sentence), then one field per question, named as "
    "the question, holding the option's value."
)


def _options(q: ChoiceQuestion | PredicateQuestion) -> list[tuple[str, str]]:
    if isinstance(q, PredicateQuestion):
        return [(YES, "the statement is true"), (NO, "the statement is false")]
    return [(c.value, c.description) for c in q.choices]


def fallback_schema(questions: Sequence[ChoiceQuestion | PredicateQuestion]) -> type[BaseModel]:
    fields: dict[str, Any] = {"reasoning": (str, ...)}
    for q in questions:
        values = tuple(v for v, _ in _options(q))
        fields[q.name] = (Literal[values], ...)  # type: ignore[valid-type]
    return create_model("Decisions", **fields)


def fallback_messages(
    text: str, questions: Sequence[ChoiceQuestion | PredicateQuestion]
) -> list[dict[str, str]]:
    lines = []
    for q in questions:
        options = "; ".join(f"{v} ({d})" if d else v for v, d in _options(q))
        lines.append(f"- {q.name}: {q.instructions} Options: {options}.")
    user = f"TEXT:\n<<<\n{text}\n>>>\n\nQUESTIONS:\n" + "\n".join(lines)
    return [{"role": "system", "content": FALLBACK_SYSTEM}, {"role": "user", "content": user}]


def fallback_decisions(
    reply: BaseModel, questions: Sequence[ChoiceQuestion | PredicateQuestion]
) -> list[Decision]:
    out = []
    for q in questions:
        value = str(getattr(reply, q.name))
        probs = {v: (1.0 if v == value else 0.0) for v, _ in _options(q)}
        out.append(
            Decision(
                name=q.name,
                type=q.type,
                choice=value,
                probabilities=probs,
                confidence=1.0,
                fallback=True,
            )
        )
    return out


def refusals(questions: Sequence[ChoiceQuestion | PredicateQuestion]) -> list[Decision]:
    return [Decision(name=q.name, type=q.type, refusal=True) for q in questions]
