#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGETS=("$ROOT_DIR")

if [[ ${1-} != "" ]]; then
  case "$1" in
    root) TARGETS=("$ROOT_DIR") ;;
    exp2|exp3|exp4) TARGETS=("$ROOT_DIR/$1") ;;
    *)
      echo "Usage: $0 [root|exp2|exp3|exp4]"
      exit 2
      ;;
  esac
fi

fail_count=0
warn_count=0

section() {
  echo
  echo "== $1 =="
}

check_syntax() {
  section "Shell Syntax"
  local ok=0
  local bad=0
  while IFS= read -r -d '' f; do
    if bash -n "$f"; then
      ((ok+=1)) || true
    else
      echo "SYNTAX_ERROR: $f"
      ((bad+=1)) || true
    fi
  done < <(find "${TARGETS[@]}" -type f -name '*.sh' -print0)

  echo "Checked: $ok OK, $bad failed"
  ((fail_count+=bad)) || true
}

check_source_targets() {
  section "Sourced Config Files"
  local missing=0
  while IFS=: read -r file _ line; do
    local src
    src="$(printf '%s' "$line" | sed -E 's/.*source[[:space:]]+\.\/([^[:space:]]+).*/\1/')"
    [[ "$src" == "$line" ]] && continue

    local basedir
    basedir="$(cd "$(dirname "$file")" && pwd)"
    local candidate="$basedir/$src"
    if [[ ! -f "$candidate" ]]; then
      echo "MISSING_SOURCE: $file -> ./$src"
      ((missing+=1)) || true
    fi
  done < <(grep -RInE '^[[:space:]]*source[[:space:]]+\./[^[:space:]]+' "${TARGETS[@]}" --include='*.sh' || true)

  if [[ $missing -eq 0 ]]; then
    echo "All sourced files exist."
  fi
  ((fail_count+=missing)) || true
}

check_conf_mutation_hotspots() {
  section "Conf Mutation Hotspots"
  local hits
  hits="$(grep -RInE '^[[:space:]]*sed -i .*conf\.sh' "${TARGETS[@]}" --include='*.sh' || true)"
  if [[ -n "$hits" ]]; then
    echo "$hits"
    echo "WARNING: scripts above mutate conf.sh in place."
    ((warn_count+=1)) || true
  else
    echo "No sed -i conf.sh patterns found."
  fi
}

check_external_binary_paths() {
  section "External Binary Paths"
  local missing=0
  while IFS=: read -r file _ token; do
    local cleaned
    cleaned="$(printf '%s' "$token" | awk '{print $1}' | sed 's/["\\]//g')"
    [[ -z "$cleaned" ]] && continue

    local basedir
    basedir="$(cd "$(dirname "$file")" && pwd)"
    local abs
    abs="$(cd "$basedir" && realpath -m "$cleaned")"

    if [[ ! -e "$abs" ]]; then
      echo "MISSING_BIN_PATH: $file -> $cleaned"
      ((missing+=1)) || true
    fi
  done < <(grep -RInE '^[[:space:]]*(\./)?\.\./[^[:space:]]*/build/[^[:space:]\\]+' "${TARGETS[@]}" --include='*.sh' || true)

  if [[ $missing -eq 0 ]]; then
    echo "External binary paths found and resolved."
  fi
  ((warn_count+=missing)) || true
}

check_conf_basics() {
  section "Conf Baseline Keys"
  local missing=0
  while IFS= read -r -d '' conf; do
    for key in dataset attr_type query_sel M ef_construction K; do
      if ! grep -qE "^${key}=" "$conf"; then
        echo "MISSING_KEY: $conf -> $key"
        ((missing+=1)) || true
      fi
    done
  done < <(find "${TARGETS[@]}" -type f -name 'conf.sh' -print0)

  if [[ $missing -eq 0 ]]; then
    echo "All conf.sh files include baseline keys."
  fi
  ((fail_count+=missing)) || true
}

check_syntax
check_source_targets
check_conf_mutation_hotspots
check_external_binary_paths
check_conf_basics

section "Summary"
echo "Failures: $fail_count"
echo "Warnings: $warn_count"

if [[ $fail_count -gt 0 ]]; then
  exit 1
fi

exit 0
