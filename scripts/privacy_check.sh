#!/usr/bin/env bash
# Pre-commit guard for a PUBLIC repo that sits next to a PRIVATE personal vault.
# Blocks a commit if staged changes contain:
#   1. files under vault-like paths or data files (sqlite, jsonl transcripts)
#   2. any term from the vault's denylist (names/aliases of private pages + manual entries)
#   3. secrets (via gitleaks, when installed)
# Run manually on the whole tree with: scripts/privacy_check.sh --all
set -euo pipefail

VAULT="${BRAIN_VAULT:-$HOME/brain-vault}"
DENYLIST="$VAULT/.privacy/denylist.txt"
mode="${1:-staged}"
fail=0

if [[ "$mode" == "--all" ]]; then
  files=$(git ls-files)
else
  files=$(git diff --cached --name-only --diff-filter=ACMR)
fi
[[ -z "$files" ]] && exit 0

# 1. Forbidden paths (fixtures and the vault template are allowed).
bad_paths=$(echo "$files" | grep -Ev '^(tests/fixtures/|vault_template/)' \
  | grep -E '(^|/)(raw|episodes|journal|reviews|\.brain|\.privacy)/|\.(sqlite3?|db|jsonl)$' || true)
if [[ -n "$bad_paths" ]]; then
  echo "privacy_check: vault-like or data files staged:" >&2
  echo "$bad_paths" | sed 's/^/  /' >&2
  fail=1
fi

# 2. Denylisted terms (case-insensitive, whole words) in added lines.
if [[ -f "$DENYLIST" ]]; then
  terms=$(grep -Ev '^\s*(#|$)' "$DENYLIST" || true)
  if [[ -n "$terms" ]]; then
    if [[ "$mode" == "--all" ]]; then
      content=$(git grep -I -n -i -w -F -f <(echo "$terms") -- . ':!tests/fixtures' || true)
    else
      content=$(git diff --cached -U0 --no-color -- . ':!tests/fixtures' \
        | grep -E '^\+[^+]' | grep -i -w -F -f <(echo "$terms") || true)
    fi
    if [[ -n "$content" ]]; then
      echo "privacy_check: denylisted personal terms found (from $DENYLIST):" >&2
      echo "$content" | head -20 | sed 's/^/  /' >&2
      fail=1
    fi
  fi
else
  echo "privacy_check: note: no denylist at $DENYLIST (run 'brain init-vault' to create one)" >&2
fi

# 3. Secrets.
if command -v gitleaks >/dev/null 2>&1; then
  if [[ "$mode" == "--all" ]]; then
    gitleaks detect --no-banner --redact --log-level warn --source . || fail=1
  else
    gitleaks protect --staged --no-banner --redact --log-level warn || fail=1
  fi
fi

if [[ $fail -ne 0 ]]; then
  echo "privacy_check: commit blocked. This repo is PUBLIC — keep personal data in the vault." >&2
  exit 1
fi
