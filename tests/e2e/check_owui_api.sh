#!/usr/bin/env bash
# Static compatibility check: does an Open WebUI release still provide every
# open_webui.* symbol, method and injected __param__ the functions in this repo use?
#
# usage: tests/e2e/check_owui_api.sh [TAG]      TAG: v0.11.4 (default) | latest | any git ref
#
# Reads pipelines/**/*.py and filters/*.py, fetches the matching Open WebUI backend
# modules at TAG (via `gh api` when authenticated, otherwise raw.githubusercontent.com
# with curl) and reports
#   - every `from open_webui.X import a, b` symbol defined (or re-exported) in X;
#     definitions marked "legacy"/"deprecated" are flagged as WARN
#   - every module imported as `import open_webui.X [as y]`
#   - every method called on an imported class object (Users.get_user_by_id(...))
#   - every __dunder__ parameter the functions declare still injected by Open WebUI
#   - frontmatter (version / required_open_webui_version / requirements) per file
#   - event types Open WebUI persists from __event_emitter__, versions of shared deps
# Exit code: 0 all ok (WARN allowed), 1 something missing, 2 fetch/usage error
# (a fetch that fails for another reason than "404 not found" is tried 3 times).
# Needs: bash, curl or gh, grep, sed, awk. Run it whenever Open WebUI publishes a
# release (see docs/testing.md); the Docker E2E suites (run.sh) cover behaviour.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
TAG=${1:-v0.11.4}
GH_REPO=open-webui/open-webui
CACHE=$(mktemp -d)
trap 'rm -rf "$CACHE"' EXIT
FAILS=0
WARNS=0

USE_GH=0
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then USE_GH=1; fi

if [ "$TAG" = latest ]; then
  if [ "$USE_GH" = 1 ]; then
    TAG=$(gh api "repos/$GH_REPO/releases/latest" --jq .tag_name || true)
  else
    TAG=$({ curl -fsSL "https://api.github.com/repos/$GH_REPO/releases/latest" || true; } \
      | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n1)
  fi
  [ -n "$TAG" ] || { echo "could not resolve the latest release" >&2; exit 2; }
fi

# fetch <path in the Open WebUI repo> -> local copy (empty file when missing).
# A failure other than "404 not found" (network, rate limit, auth) is retried
# (FETCH_TRIES attempts in all); when it persists it is recorded in
# $FETCH_ERRORS and the script ends with exit 2 instead of reporting the module
# as missing. fetch runs in $(...) subshells, hence the file.
FETCH_ERRORS="$CACHE/.fetch_errors"
FETCH_TRIES=3
fetch() {
  local path=$1 dest="$CACHE/$1" code err try
  if [ ! -e "$dest" ]; then
    mkdir -p "$(dirname "$dest")"
    for ((try = 1; try <= FETCH_TRIES; try++)); do
      [ "$try" = 1 ] || sleep $((try * 2))
      err=""
      if [ "$USE_GH" = 1 ]; then
        gh api -H "Accept: application/vnd.github.raw" \
          "repos/$GH_REPO/contents/$path?ref=$TAG" >"$dest" 2>"$dest.err" && break
        if grep -q "HTTP 404" "$dest.err"; then : >"$dest"; break; fi
        err=$(tr '\n' ' ' <"$dest.err" | cut -c1-160)
      else
        code=$(curl -sSL -o "$dest" -w '%{http_code}' \
          "https://raw.githubusercontent.com/$GH_REPO/$TAG/$path" 2>/dev/null || true)
        [ "$code" != 200 ] || break
        if [ "$code" = 404 ]; then : >"$dest"; break; fi
        err="HTTP ${code:-error}"
      fi
    done
    rm -f "$dest.err"
    if [ -n "$err" ]; then
      echo "$path: $err ($FETCH_TRIES attempts)" >>"$FETCH_ERRORS"
      : >"$dest"
    fi
  fi
  echo "$dest"
}
fetch_failed() { [ -s "$FETCH_ERRORS" ] && grep -q "^$1" "$FETCH_ERRORS"; }

# module_file open_webui.models.users -> local copy of models/users.py (or __init__.py)
module_file() {
  local rel f
  rel=backend/$(echo "$1" | tr . /)
  f=$(fetch "$rel.py")
  [ -s "$f" ] || f=$(fetch "$rel/__init__.py")
  echo "$f"
}

ok() { printf '  ok    %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; WARNS=$((WARNS + 1)); }
bad() { printf '  FAIL  %s\n' "$*"; FAILS=$((FAILS + 1)); }

FILES=$(cd "$REPO" && ls pipelines/*/*.py filters/*.py)
[ -s "$(fetch backend/open_webui/main.py)" ] \
  || { echo "cannot fetch Open WebUI $TAG (tag exists? network?)" >&2; exit 2; }
echo "Open WebUI $TAG vs $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo worktree)"

# -- 1. imports: "module<TAB>name<TAB>file" (multi-line parenthesised imports joined)
IMPORTS=$(for f in $FILES; do
  awk -v file="$f" '
    function emit(line,   mod, names, n, i, parts, name) {
      sub(/^[ \t]*from[ \t]+/, "", line)
      mod = line; sub(/[ \t]+import[ \t].*/, "", mod)
      names = line; sub(/^[^ \t]+[ \t]+import[ \t]+/, "", names)
      gsub(/[()\\]/, " ", names); gsub(/#[^,]*/, "", names)
      n = split(names, parts, ",")
      for (i = 1; i <= n; i++) {
        name = parts[i]; sub(/[ \t]+as[ \t].*/, "", name); gsub(/[ \t\r]/, "", name)
        if (name != "") print mod "\t" name "\t" file
      }
    }
    collecting { buf = buf " " $0; if ($0 ~ /\)/) { emit(buf); collecting = 0 }; next }
    /^[ \t]*from[ \t]+open_webui[.a-zA-Z_]*[ \t]+import/ {
      if ($0 ~ /\(/ && $0 !~ /\)/) { collecting = 1; buf = $0; next }
      emit($0)
    }' "$REPO/$f"
done | tr -d '\r' | sort -u)
# unique module/name pairs with the files using them
SYMBOLS=$(echo "$IMPORTS" | awk -F'\t' 'NF == 3 {
    key = $1 "\t" $2; users[key] = (key in users) ? users[key] "," $3 : $3 }
  END { for (k in users) print k "\t" users[k] }' | sort)

echo
echo "== open_webui imports"
B='([^A-Za-z0-9_]|$)'
while IFS=$'\t' read -r mod name users; do
  [ -n "$name" ] || continue
  src=$(module_file "$mod")
  if [ ! -s "$src" ]; then
    if fetch_failed "backend/$(echo "$mod" | tr . /)"; then
      printf '  ERROR %s could not be fetched (see the end of the output)\n' "$mod"
    else
      bad "$mod (module missing) <- $users"
    fi
    continue
  fi
  def=$(grep -Em1 "^(async[[:space:]]+)?def[[:space:]]+${name}[[:space:]]*\(|^class[[:space:]]+$name$B|^[[:space:]]*${name}[[:space:]]*(:[^=]*)?=([^=]|$)|^[[:space:]]*(from[[:space:]].*)?import[[:space:]].*[[:space:],(]$name$B|^[[:space:]]+$name,?[[:space:]]*$" "$src" || true)
  if [ -z "$def" ]; then
    bad "$mod.$name <- $users"
  elif echo "$def" | grep -qiE "legacy|deprecated"; then
    warn "$mod.$name is marked legacy/deprecated: $(echo "$def" | sed 's/^[[:space:]]*//' | cut -c1-90) <- $users"
  else
    ok "$mod.$name"
  fi
done <<<"$SYMBOLS"

# -- 1b. plain imports: "import open_webui.X [as y], ..." -> "module<TAB>files"
MODULES=$(for f in $FILES; do
  { grep -E '^[[:space:]]*import[[:space:]]' "$REPO/$f" || true; } | tr -d '\r' \
    | sed -E 's/#.*//; s/^[[:space:]]*import[[:space:]]+//' | tr ',' '\n' \
    | sed -E 's/[[:space:]]+as[[:space:]].*//; s/[[:space:]]//g' \
    | { grep -E '^open_webui([.]|$)' || true; } | awk -v file="$f" '{ print $0 "\t" file }'
done | sort -u | awk -F'\t' 'NF == 2 {
    users[$1] = ($1 in users) ? users[$1] "," $2 : $2 }
  END { for (m in users) print m "\t" users[m] }' | sort)

echo
echo "== open_webui modules imported with 'import'"
[ -n "$MODULES" ] || echo "  (none)"
while IFS=$'\t' read -r mod users; do
  [ -n "$mod" ] || continue
  if [ -s "$(module_file "$mod")" ]; then
    ok "$mod (module)"
  elif fetch_failed "backend/$(echo "$mod" | tr . /)"; then
    printf '  ERROR %s could not be fetched (see the end of the output)\n' "$mod"
  else
    bad "$mod (module missing) <- $users"
  fi
done <<<"$MODULES"

# -- 2. methods called on imported class objects, e.g. Users.get_user_by_id(
echo
echo "== methods called on imported objects"
while IFS=$'\t' read -r mod name users; do
  case "$name" in [A-Z][a-z]*) ;; *) continue ;; esac
  src=$(module_file "$mod")
  [ -s "$src" ] || continue
  # shellcheck disable=SC2046 # the file list is split into words on purpose
  for method in $(cd "$REPO" && cat $(echo "$users" | tr , ' ') \
      | grep -oE "(^|[^A-Za-z0-9_.])$name\.[a-z_][a-z0-9_]*\(" \
      | sed -E "s/.*$name\.([a-z0-9_]+)\(/\1/" | sort -u); do
    if grep -Eq "^[[:space:]]*(async[[:space:]]+)?def[[:space:]]+${method}[[:space:]]*\(" "$src"; then
      ok "$name.$method() ($mod)"
    else
      bad "$name.$method() not found in $mod <- $users"
    fi
  done
done <<<"$SYMBOLS"

# -- 3. injected __params__ declared by pipe()/inlet()/outlet()/stream()
echo
echo "== injected __params__"
# (grep finding nothing must not end the script under set -e / pipefail)
PROVIDED=$(cat "$(fetch backend/open_webui/functions.py)" \
  "$(fetch backend/open_webui/utils/filter.py)" \
  "$(fetch backend/open_webui/utils/middleware.py)" \
  | { grep -oE "['\"]__[a-z][a-z_]*__['\"]" || true; } | tr -d "'\"" | sort -u)
for param in $(cd "$REPO" && cat $FILES \
    | grep -oE "^[[:space:]]*(self, )?__[a-z][a-z_]*__[[:space:]]*(:|=|,|\))" \
    | grep -oE "__[a-z][a-z_]*__" | sort -u); do
  users=$(cd "$REPO" && grep -lE "^[[:space:]]*(self, )?${param}[[:space:]]*(:|=|,|\))" $FILES | tr '\n' ' ')
  if echo "$PROVIDED" | grep -qx "$param"; then ok "$param"; else bad "$param not injected any more <- $users"; fi
done

# -- 4. frontmatter as Open WebUI parses it (first docstring, "key: value" lines)
echo
echo "== frontmatter (version / required_open_webui_version / requirements)"
for f in $FILES; do
  awk -v file="$f" 'NR == 1 { if ($0 !~ /^[ \t]*"""[ \t\r]*$/) exit; next }
    /"""/ { exit }
    /^(version|required_open_webui_version|requirements):/ {
      sub(/\r$/, ""); out = out "  " $0 }
    END { print "  " file ":" out }' "$REPO/$f"
done

# -- 5. informational: persisted emitter events, shared dependency versions
#       (never fail the check; "none found" just means the code moved)
echo
echo "== event types persisted by __event_emitter__ at $TAG"
events=$({ grep -oE "event_type (==|in) (\([^)]*\)|'[^']*')" \
  "$(fetch backend/open_webui/socket/main.py)" || true; } \
  | sed -E "s/event_type (==|in) //" | tr -d "()'" | tr ',' '\n' | sed 's/^ *//' \
  | sort -u | tr '\n' ' ')
echo "  ${events:-(none found in backend/open_webui/socket/main.py)}"
echo
echo "== shared dependencies at $TAG"
deps=$(cat "$(fetch pyproject.toml)" "$(fetch backend/requirements.txt)" \
  "$(fetch backend/requirements-slim.txt)" 2>/dev/null \
  | { grep -ioE "^[[:space:]]*\"?(aiohttp|aiofiles|cryptography|pydantic|fastapi|pillow|google-genai|httpx|python-socketio)[=<>~!][^\",#[:space:]]*" || true; } \
  | tr -d ' "' | sort -u | tr '\n' ' ')
echo "  ${deps:-(none found)}"

echo
if [ -s "$FETCH_ERRORS" ]; then
  echo "could not fetch from Open WebUI $TAG (network, rate limit or auth?):"
  sed 's/^/  /' "$FETCH_ERRORS"
  echo "RESULT: incomplete, $FAILS missing API(s) among the files that could be fetched"
  exit 2
fi
if [ "$FAILS" -gt 0 ]; then
  echo "RESULT: $FAILS missing API(s), $WARNS warning(s) at Open WebUI $TAG"
  exit 1
fi
echo "RESULT: all open_webui APIs used by the functions exist at Open WebUI $TAG ($WARNS warning(s))"
