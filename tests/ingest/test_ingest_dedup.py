"""Paraphrased duties, skills and interests merge (``brain.ingest.dedup``), conservatively."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import httpx
import numpy as np
import pytest

from brain.ingest.dedup import ASK_ABOVE, INSTRUCT, MERGE_ABOVE, merge_paraphrases, words
from brain.ingest.normalise import template_text
from brain.llm.decide import Decision, Question
from brain.vault.ops import AddFact, ChangeSet, CreateEntity
from brain.vault.store import Vault
from brain.vault.writer import apply

SRC = "raw/docs/2026-10-25-alex-rivera.html"


def vector(cos: float, axis: int = 1) -> np.ndarray:
    """A unit vector at cosine ``cos`` from ``e0`` (``axis`` picks the orthogonal part)."""
    v = np.zeros(8, dtype=np.float32)
    v[0], v[axis] = cos, np.sqrt(1 - cos**2)
    return v


class Scripted:
    """Embeddings by statement, decisions by pair; records what was asked."""

    def __init__(
        self,
        vectors: dict[str, np.ndarray],
        same: dict[tuple[str, str], float] | None = None,
        embed_error: Exception | None = None,
    ) -> None:
        self.vectors, self.same, self.embed_error = vectors, same or {}, embed_error
        self.asked: list[str] = []

    def embed(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:
        if self.embed_error:
            raise self.embed_error
        assert all(t.startswith(INSTRUCT) for t in texts)
        out = []
        for n, text in enumerate(texts):
            statement = text.removeprefix(INSTRUCT)
            out.append(self.vectors.get(statement, vector(0.0, axis=2 + n % 6)))
        return np.stack(out)

    def decide(self, text: str, questions: Sequence[Question]) -> list[Decision]:
        self.asked.append(text)
        a, b = (line.split(": ", 1)[1] for line in text.splitlines())
        p = self.same.get((a, b), self.same.get((b, a), 0.0))
        return [
            Decision(
                name="same",
                type="choice",
                choice="same" if p >= 0.5 else "different",
                probabilities={"same": p, "different": 1 - p},
                confidence=abs(2 * p - 1),
            )
        ]


def duty(o: str, text: str = "", start: str | None = None, end: str | None = None) -> AddFact:
    text = text or (o if o[:1].isupper() else f"Responsible for {o}")
    return AddFact(
        subject="self",
        p="responsibility",
        o=o,
        text=text,
        kind="stated",
        valid_from=start,
        valid_to=end,
    )


def changeset(*ops) -> ChangeSet:
    return ChangeSet(ops=list(ops), origin="ingest", src=[SRC])


def facts(cs: ChangeSet) -> list[AddFact]:
    return [op for op in cs.ops if isinstance(op, AddFact)]


FINE_TUNE = "Fine-tunes graph neural networks for solubility prediction"
FINE_TUNE_2 = "fine-tuning graph neural networks to predict solubility"
RANKING = "Builds models that help chemists rank new molecules"
RANKING_2 = "Helps chemists pick which molecules to make"


def test_close_paraphrases_merge_keeping_the_more_specific_and_both_periods(
    fresh_vault: Vault,
) -> None:
    llm = Scripted({FINE_TUNE: vector(1.0), FINE_TUNE_2: vector(0.95)})
    cs, report = merge_paraphrases(
        changeset(
            duty(FINE_TUNE_2, start="2023-03"),
            duty("graph neural networks", FINE_TUNE, start="2022", end="2024"),
        ),
        fresh_vault,
        llm,
    )
    (kept,) = facts(cs)
    assert kept.text == FINE_TUNE  # more words
    assert (kept.valid_from, kept.valid_to) == ("2023-03", None)  # precise start, still open
    assert report.merged == 1 and llm.asked == []
    assert report.notes == [
        f"dedup: merged responsibility {FINE_TUNE!r} into {FINE_TUNE_2!r} (cosine 0.95), "
        f"kept {FINE_TUNE!r}"
    ]


def test_borderline_pairs_are_asked(fresh_vault: Vault) -> None:
    cos = (ASK_ABOVE + MERGE_ABOVE) / 2
    other = "Builds models that predict antibody stability"
    vectors = {RANKING: vector(1.0), RANKING_2: vector(cos), other: vector(cos, axis=3)}
    llm = Scripted(vectors, same={(RANKING, RANKING_2): 0.93, (RANKING, other): 0.1})
    cs, report = merge_paraphrases(
        changeset(duty(RANKING), duty(RANKING_2), duty(other)), fresh_vault, llm
    )
    assert [f.o for f in facts(cs)] == [RANKING, other]
    assert report.asked == 2 and report.merged == 1  # other: asked about the kept one only
    assert "p(same)=0.93" in report.notes[0]


def test_unsure_or_distant_pairs_stay_apart(fresh_vault: Vault) -> None:
    llm = Scripted({RANKING: vector(1.0), RANKING_2: vector(0.8)}, same={(RANKING, RANKING_2): 0.6})
    cs, _ = merge_paraphrases(changeset(duty(RANKING), duty(RANKING_2)), fresh_vault, llm)
    assert len(facts(cs)) == 2
    llm = Scripted({RANKING: vector(1.0), RANKING_2: vector(ASK_ABOVE - 0.05)})
    cs, report = merge_paraphrases(changeset(duty(RANKING), duty(RANKING_2)), fresh_vault, llm)
    assert len(facts(cs)) == 2 and report.asked == 0
    # asking switched off: only the embeddings decide
    llm = Scripted({RANKING: vector(1.0), RANKING_2: vector(0.8)}, same={(RANKING, RANKING_2): 1})
    cs, _ = merge_paraphrases(
        changeset(duty(RANKING), duty(RANKING_2)), fresh_vault, llm, ask=False
    )
    assert len(facts(cs)) == 2 and llm.asked == []


def test_same_duty_at_two_jobs_is_not_merged(fresh_vault: Vault) -> None:
    llm = Scripted({RANKING: vector(1.0), RANKING_2: vector(0.99)})
    cs, _ = merge_paraphrases(
        changeset(duty(RANKING, start="2018", end="2019"), duty(RANKING_2, start="2022")),
        fresh_vault,
        llm,
    )
    assert len(facts(cs)) == 2


def skill(o: str, p: str = "has_skill") -> AddFact:
    text = template_text(p, o, past=False)
    return AddFact(subject="self", p=p, o=o, text=text, kind="stated")


def test_same_words_merge_without_the_embedding_endpoint(fresh_vault: Vault) -> None:
    down = Scripted({}, embed_error=httpx.ConnectError("refused"))
    cs, report = merge_paraphrases(
        changeset(skill("deep-learning"), skill("Deep learning"), skill("Python")),
        fresh_vault,
        down,
    )
    assert [f.o for f in facts(cs)] == ["deep-learning", "Python"]
    assert report.notes == [
        "dedup: no embeddings (ConnectError): paraphrases not merged",
        "dedup: merged has_skill 'Deep learning' into 'deep-learning' (same words)",
    ]
    assert words("Accessibility (WCAG)") == ("accessibility", "wcag")


def test_only_multi_valued_literals_of_one_subject_and_predicate(fresh_vault: Vault) -> None:
    llm = Scripted({"Python": vector(1.0)})
    ops = [
        skill("Python"),
        skill("Python", p="interest"),  # another predicate
        AddFact(subject="projects/atlas", p="uses", o="Python", text="Uses Python", kind="stated"),
        AddFact(subject="self", p="works_at", o="[[orgs/northwind-labs]]", text="", kind="stated"),
    ]
    cs, report = merge_paraphrases(changeset(*ops), fresh_vault, llm)
    assert cs.ops == ops and report.merged == 0


def test_a_restatement_of_a_vault_fact_takes_its_value(tmp_vault: Vault) -> None:
    """The fixture vault has "Keeps the nightly batch pipelines running" (o: Nightly batch
    pipelines); the new wording becomes that fact, and the writer skips it as a duplicate."""
    old = "Keeps the nightly batch pipelines running"
    new = "Maintains the nightly batch pipelines"
    llm = Scripted({old: vector(1.0), new: vector(0.96)})
    cs, report = merge_paraphrases(changeset(duty(new)), tmp_vault, llm)
    (fact,) = facts(cs)
    assert (fact.o, fact.text) == ("Nightly batch pipelines", old)
    assert report.notes == [
        f"dedup: responsibility {new!r} restates the vault's {old!r} (cosine 0.96)"
    ]
    result = apply(cs, tmp_vault, now=datetime(2026, 10, 25, 12))
    assert result.applied == [] and result.skipped


def test_nothing_to_compare_makes_no_call(fresh_vault: Vault) -> None:
    llm = Scripted({}, embed_error=AssertionError("no call expected"))
    cs = changeset(CreateEntity(id="orgs/x", type="org", title="X"), skill("Python"))
    out, report = merge_paraphrases(cs, fresh_vault, llm)
    assert out.ops == cs.ops and report.embedded == 0


@pytest.mark.parametrize("p", ["interest", "habit", "likes", "goal", "values", "certification"])
def test_other_predicates_merge_too(fresh_vault: Vault, p: str) -> None:
    llm = Scripted({"bouldering": vector(1.0), "indoor bouldering": vector(0.97)})
    ops = [skill("bouldering", p=p), skill("indoor bouldering", p=p)]
    cs, _ = merge_paraphrases(changeset(*ops), fresh_vault, llm)
    assert [f.o for f in facts(cs)] == ["indoor bouldering"]
