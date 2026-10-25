"""``decide``: the ``/v1/decisions`` client, its structured fallback, the fake and the cache."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from brain.config import LLMSettings
from brain.llm import LLM, ChoiceQuestion, Decision, PredicateQuestion
from brain.llm.cache import StructuredCache
from brain.llm.decide import Choice
from brain.llm.fake import FakeLLM
from brain.llm.recorder import Recorder

KIND = ChoiceQuestion(
    name="kind",
    instructions='What kind of thing is "Airflow" in the text?',
    choices=[Choice(value="tool", description="software or product"), Choice(value="project")],
)
FACTS = PredicateQuestion(name="facts", instructions="Does the text say anything about Alex?")
TEXT = "Alex Rivera leads the Atlas Migration, which uses Airflow."
ANSWERS = {
    "object": "decision",
    "answers": [
        {
            "type": "choice",
            "name": "kind",
            "choice": "tool",
            "probabilities": [
                {"value": "tool", "probability": 0.9006},
                {"value": "project", "probability": 0.0994},
            ],
            "confidence": 0.8012,
        },
        {"type": "predicate", "name": "facts", "probability": 0.9627},
    ],
    "usage": {"input_tokens": 256, "output_tokens": 0},
}


def completion(content: str) -> dict:
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "chat-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def client(handler, **kw) -> LLM:
    return LLM(LLMSettings(), transport=httpx.MockTransport(handler), **kw)


def test_decide_posts_typed_questions_and_parses_answers(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ANSWERS)

    llm = client(handler, recorder=Recorder(tmp_path))
    kind, facts = llm.decide(TEXT, [KIND, FACTS])
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/decisions"
    assert body["model"] == LLMSettings().chat_model and body["input"] == TEXT
    assert body["questions"][0]["choices"][0] == {
        "value": "tool",
        "description": "software or product",
    }
    assert body["questions"][1] == {
        "type": "predicate",
        "name": "facts",
        "instructions": FACTS.instructions,
    }
    assert kind.choice == "tool" and kind.confidence == 0.8012 and not kind.fallback
    assert kind.probabilities == {"tool": 0.9006, "project": 0.0994}
    assert facts.p_yes == 0.9627 and facts.choice == "yes" and facts.probabilities["no"] == 0.0373
    [saved] = tmp_path.glob("decide-*.json")
    assert json.loads(saved.read_text())["kind"] == "decide"
    assert llm.decide(TEXT, []) == []


def test_refusal_and_missing_answers() -> None:
    reply = {"answers": [{"type": "refusal", "name": "kind"}]}
    llm = client(lambda r: httpx.Response(200, json=reply))
    kind, facts = llm.decide(TEXT, [KIND, FACTS])
    assert kind.refusal and kind.choice is None and kind.p_yes is None
    assert facts.refusal  # not answered at all
    assert "refusal" in kind.summary()


def test_no_decisions_endpoint_falls_back_to_one_structured_call() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/decisions"):
            return httpx.Response(404, json={"detail": "Not Found"})
        body = json.loads(request.content)
        schema = body["response_format"]["json_schema"]["schema"]
        names = list(schema["properties"])
        assert names[0] == "reasoning" and names[-1] == "facts"  # reasoning first
        assert schema["properties"]["facts"]["enum"] == ["yes", "no"]
        assert body["temperature"] == 0.0
        if "kind" not in names:
            return httpx.Response(200, json=completion('{"reasoning": "r", "facts": "yes"}'))
        assert schema["properties"]["kind"]["enum"] == ["tool", "project"]
        assert "Airflow" in body["messages"][1]["content"]
        return httpx.Response(
            200, json=completion('{"reasoning": "r", "kind": "tool", "facts": "no"}')
        )

    llm = client(handler)
    kind, facts = llm.decide(TEXT, [KIND, FACTS])
    assert kind.fallback and kind.choice == "tool" and kind.confidence == 1.0
    assert kind.probabilities == {"tool": 1.0, "project": 0.0}
    assert facts.fallback and facts.p_yes == 0.0
    llm.decide(TEXT, [FACTS])  # remembered: no second probe of /v1/decisions
    assert paths == ["/v1/decisions", "/v1/chat/completions", "/v1/chat/completions"]


def test_api_errors_are_not_mistaken_for_a_missing_endpoint() -> None:
    error = {"error": {"message": "Model 'nope' not found", "type": "not_found_error"}}
    llm = client(lambda r: httpx.Response(404, json=error))
    with pytest.raises(httpx.HTTPStatusError):
        llm.decide(TEXT, [FACTS])
    llm = client(lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(httpx.HTTPStatusError):
        llm.decide(TEXT, [FACTS])


def test_fake_scripts_decisions_by_name_queue_and_decider(tmp_path: Path) -> None:
    yes = Decision(
        name="facts", type="predicate", choice="yes", probabilities={"yes": 0.9, "no": 0.1}
    )
    fake = FakeLLM(decisions=[[yes], ANSWERS, RuntimeError("down")])
    kind, facts = fake.decide(TEXT, [KIND, FACTS])
    assert kind.refusal and facts.p_yes == 0.9  # unscripted names are refusals
    kind, facts = fake.decide(TEXT, [KIND, FACTS])  # a recorded /v1/decisions response
    assert kind.choice == "tool" and facts.p_yes == 0.9627
    with pytest.raises(RuntimeError):
        fake.decide(TEXT, [FACTS])
    [refused] = fake.decide(TEXT, [FACTS])  # nothing scripted: no answer
    assert refused.refusal
    assert fake.calls[0] == ("decide", {"text": TEXT, "questions": [KIND, FACTS]})
    fake = FakeLLM(decider=lambda text, qs: [yes])
    assert fake.decide(TEXT, [FACTS])[0].p_yes == 0.9

    path = Recorder(tmp_path).save("decide", {"input": TEXT}, ANSWERS)
    replay = FakeLLM.from_fixtures([path])
    assert replay.decide(TEXT, [KIND, FACTS])[0].choice == "tool"


def test_cache_replays_decisions_but_not_refusals(tmp_path: Path) -> None:
    fake = FakeLLM(decisions=[ANSWERS, {"answers": []}, {"answers": []}])
    llm = StructuredCache(fake, tmp_path)
    first = llm.decide(TEXT, [KIND, FACTS])
    assert llm.decide(TEXT, [KIND, FACTS]) == first and llm.hits == 1
    assert len([c for c in fake.calls if c[0] == "decide"]) == 1
    llm.decide("other text", [FACTS])  # a refusal is not cached: asked again next time
    llm.decide("other text", [FACTS])
    assert len([c for c in fake.calls if c[0] == "decide"]) == 3
