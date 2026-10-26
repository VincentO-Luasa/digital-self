"""Generated files never show questions.md items marked ``(privacy:: private)``."""

from __future__ import annotations

from datetime import date

from brain.vault.generated import build_core
from brain.vault.store import Vault

QUESTIONS = """# Questions

- [ ] Where does Atlas Migration stand right now? (region:: projects)
  (about:: [[projects/atlas-migration]]) (slot:: status) ^q-a2b3c4
- [ ] How is Sam doing these days? (region:: relationships.friends)
  (about:: [[people/sam-okafor]]) (privacy:: private) ^q-d5e6f7
- [ ] What tends to trip you up? (region:: self.traits) (privacy:: private) ^q-g2h3i4
"""


def test_private_questions_are_not_shown(tmp_vault: Vault) -> None:
    tmp_vault.questions_path.write_text(QUESTIONS)
    for include_private in (True, False):
        core = build_core(tmp_vault, today=date(2026, 10, 27), include_private=include_private)
        assert "To ask: Where does Atlas Migration stand right now?" in core
        assert "trip you up" not in core
        assert "How is Sam" not in core
