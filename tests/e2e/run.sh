#!/usr/bin/env bash
# Local end-to-end tests for the Open WebUI functions in this repo.
#
# Starts a throw-away Open WebUI container, copies the function files, the
# provider mocks and the driver into it, runs the scenario suites INSIDE the
# container and removes container + volume again. Host needs: Docker and bash
# (Linux, macOS, Git Bash on Windows). No host Python.
#
# Docs: docs/testing.md
set -euo pipefail

# Git Bash on Windows rewrites arguments that look like POSIX paths
# (/e2e -> C:/Program Files/Git/e2e). Every docker call goes through dk(), which
# disables that rewriting so container paths stay untouched; host files are
# streamed with tar, so docker.exe never gets a host path. Other tools (git, tar)
# need the normal conversion for /c/... host paths, so a globally exported
# MSYS_NO_PATHCONV / MSYS2_ARG_CONV_EXCL is dropped here.
unset MSYS_NO_PATHCONV MSYS2_ARG_CONV_EXCL
dk() { MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' docker "$@"; }

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

# Suites are the modules in tests/e2e/suites/ (e2e.py also checks harness/config.py).
available_suites() {
  local f
  for f in "$HERE"/suites/*.py; do
    f=${f##*/}
    f=${f%.py}
    [ "$f" = __init__ ] || printf '%s ' "$f"
  done
}

usage() {
  echo "usage: tests/e2e/run.sh [options] [suite ...]"
  echo
  echo "suites: $(available_suites)all (default: all)"
  cat <<'EOF'

options:
  -i, --image IMAGE    Open WebUI image or tag (default: $OWUI_IMAGE or
                       ghcr.io/open-webui/open-webui:v0.11.4-slim); a bare tag
                       like v0.11.3-slim means ghcr.io/open-webui/open-webui:TAG
  -s, --suites LIST    comma separated suites (same as positional suites)
      --only REGEX     run only scenario groups matching REGEX on "<suite>.<group>"
                       (Python regex; alternatives as 'gemini.(api|image)')
      --ref REF        test the function files as of git REF (git show REF:path)
      --src DIR        test the function files from DIR (another checkout/worktree)
      --file PATH=FILE use FILE for repo path PATH (repeatable), e.g.
                       --file pipelines/azure/azure_ai_foundry.py=/tmp/fix.py
                       (PATH must be one of the function files listed below)
  -n, --name NAME      container name (default: $E2E_NAME or owui-e2e-<random>);
                       the volume is NAME-data
  -p, --port PORT      host port for the UI on 127.0.0.1 (default: $E2E_PORT or
                       a free port chosen by Docker)
  -o, --out DIR        output directory (default: tests/e2e/out/<timestamp>-<name>)
  -k, --keep           keep container + volume running afterwards (debugging)
      --reuse          reuse the running container NAME (implies --keep): skips
                       start-up, re-copies files, restarts the mocks
  -v, --verbose        print details of passing scenarios too
      --strict-known   a check tagged with a known bug that passes while its
                       marker still applies is a FAIL (default: $E2E_STRICT_KNOWN)
  -h, --help           this help

exit code: 0 = only PASS/KNOWN, 1 = at least one FAIL, 2 = setup error
EOF
  echo
  echo "function files: ${FUNCTION_FILES[*]}"
}

IMAGE=${OWUI_IMAGE:-ghcr.io/open-webui/open-webui:v0.11.4-slim}
NAME=${E2E_NAME:-}
PORT=${E2E_PORT:-}
OUT=${E2E_OUT:-}
KEEP=${E2E_KEEP:-0}
REUSE=0
ONLY=""
REF=""
SRC_DIR=""
VERBOSE=""
STRICT=${E2E_STRICT_KNOWN:-}
SUITES=""
HEALTH_TIMEOUT=${E2E_HEALTH_TIMEOUT:-600}
SECRET_KEY=${E2E_SECRET_KEY:-e2e-local-secret-key-0123456789abcdef}
VERTEX_RAG_STORE="projects/e2e/locations/global/collections/default_collection/dataStores/e2e-store"
OVERRIDES=()

FUNCTION_FILES=(
  pipelines/google/google_gemini.py
  pipelines/azure/azure_ai_foundry.py
  pipelines/n8n/n8n.py
  pipelines/infomaniak/infomaniak.py
  filters/google_search_tool.py
  filters/vertex_ai_search_tool.py
  filters/time_token_tracker.py
)

die() { echo "error: $*" >&2; exit 2; }
need() { [ $# -ge 2 ] || die "option $1 needs a value (see --help)"; }
add_suites() { SUITES="${SUITES:+$SUITES,}$1"; }
# repo path as given to --file: backslashes -> slashes, no leading ./
norm_path() { local p=${1//\\//}; echo "${p#./}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    -i|--image) need "$@"; IMAGE=$2; shift 2 ;;
    -s|--suites) need "$@"; add_suites "$2"; shift 2 ;;
    --only) need "$@"; ONLY=$2; shift 2 ;;
    --ref) need "$@"; REF=$2; shift 2 ;;
    --src) need "$@"; SRC_DIR=$2; shift 2 ;;
    --file) need "$@"; OVERRIDES+=("$2"); shift 2 ;;
    -n|--name) need "$@"; NAME=$2; shift 2 ;;
    -p|--port) need "$@"; PORT=$2; shift 2 ;;
    -o|--out) need "$@"; OUT=$2; shift 2 ;;
    -k|--keep) KEEP=1; shift ;;
    --reuse) REUSE=1; KEEP=1; shift ;;
    -v|--verbose) VERBOSE=--verbose; shift ;;
    --strict-known) STRICT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) usage >&2; die "unknown option $1" ;;
    *) add_suites "$1"; shift ;;
  esac
done

# ---------------------------------------------------------------- validate
case "$STRICT" in 1 | true | yes | on) STRICT=1 ;; *) STRICT="" ;; esac
case "$IMAGE" in */*) ;; *) IMAGE="ghcr.io/open-webui/open-webui:$IMAGE" ;; esac
SUITES=${SUITES:-all}
case ",$SUITES," in *,all,*) SUITES=all ;; esac
for suite in ${SUITES//,/ }; do
  case "$suite" in
    all) ;;
    __init__ | *[!a-z0-9_]*) die "unknown suite '$suite' (choose from: $(available_suites)all)" ;;
    *) [ -f "$HERE/suites/$suite.py" ] \
         || die "unknown suite '$suite' (choose from: $(available_suites)all)" ;;
  esac
done
command -v docker >/dev/null || die "docker not found"
[ -z "$REF" ] || [ -z "$SRC_DIR" ] || die "--ref and --src exclude each other"
[ -z "$REF" ] || git -C "$REPO" rev-parse --verify --quiet "$REF^{commit}" >/dev/null \
  || die "unknown git ref $REF"
[ -z "$SRC_DIR" ] || [ -d "$SRC_DIR" ] || die "--src $SRC_DIR is not a directory"
for override in "${OVERRIDES[@]+"${OVERRIDES[@]}"}"; do
  case "$override" in
    ?*=?*) ;;
    *) die "--file needs PATH=FILE, got '$override'" ;;
  esac
  wanted=$(norm_path "${override%%=*}")
  match=""
  for path in "${FUNCTION_FILES[@]}"; do
    [ "$wanted" != "$path" ] || match=$path
  done
  [ -n "$match" ] || die "--file: unknown path '${override%%=*}' (known: ${FUNCTION_FILES[*]})"
  [ -f "${override#*=}" ] || die "--file $override: file not found"
done
NAME=${NAME:-owui-e2e-$(date +%H%M%S)-$RANDOM}
VOLUME="$NAME-data"
OUT=${OUT:-$HERE/out/$(date +%Y%m%d-%H%M%S)-$NAME}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

# ---------------------------------------------------------------- stage files
# Function files are staged under $OUT/functions (also a record of what ran).
stage_functions() {
  local staged="$OUT/functions" path src override found
  rm -rf "$staged"
  mkdir -p "$staged"
  : >"$staged/SOURCES.txt"
  for path in "${FUNCTION_FILES[@]}"; do
    mkdir -p "$staged/$(dirname "$path")"
    found=""
    for override in "${OVERRIDES[@]+"${OVERRIDES[@]}"}"; do
      if [ "$(norm_path "${override%%=*}")" = "$path" ]; then found=${override#*=}; fi
    done
    if [ -n "$found" ]; then
      cp "$found" "$staged/$path"
      src="file $found"
    elif [ -n "$REF" ]; then
      git -C "$REPO" show "$REF:$path" >"$staged/$path" || die "git show $REF:$path failed"
      src="git $REF ($(git -C "$REPO" rev-parse --short "$REF"))"
    elif [ -n "$SRC_DIR" ]; then
      [ -f "$SRC_DIR/$path" ] || die "$SRC_DIR/$path not found"
      cp "$SRC_DIR/$path" "$staged/$path"
      src="dir $SRC_DIR"
    else
      cp "$REPO/$path" "$staged/$path"
      src="worktree $REPO"
    fi
    printf '%s\t%s\n' "$path" "$src" >>"$staged/SOURCES.txt"
  done
}

# ---------------------------------------------------------------- container
CREATED=0
cleanup() {
  local rc=$?
  set +e
  if dk container inspect "$NAME" >/dev/null 2>&1; then
    dk logs "$NAME" >"$OUT/server.log" 2>&1
    if [ "$KEEP" = 1 ] && [ "$(dk inspect -f '{{.State.Running}}' "$NAME")" = true ]; then
      echo "kept container $NAME (volume $VOLUME): http://localhost:$(host_port)" \
        " login admin@example.com / Passw0rd!e2e"
      echo "  rerun:   tests/e2e/run.sh --reuse --name $NAME ..."
      echo "  cleanup: docker rm -f $NAME && docker volume rm $VOLUME"
    elif [ "$CREATED" = 1 ]; then
      dk rm -f "$NAME" >/dev/null
      dk volume rm "$VOLUME" >/dev/null
    fi
  fi
  echo "output: $OUT"
  exit "$rc"
}

host_port() { dk port "$NAME" 8080/tcp 2>/dev/null | head -n1 | sed 's/.*://'; }

start_container() {
  if dk container inspect "$NAME" >/dev/null 2>&1; then
    die "container $NAME already exists (use --reuse, or remove it)"
  fi
  dk volume rm "$VOLUME" >/dev/null 2>&1 || true
  echo "starting $NAME from $IMAGE"
  CREATED=1
  # The server output is mirrored to /tmp/e2e/server.log so the driver can
  # inspect it per scenario; `docker logs` keeps working as usual.
  dk run -d --name "$NAME" \
    -p "127.0.0.1:${PORT}:8080" \
    -v "$VOLUME:/app/backend/data" \
    -e WEBUI_SECRET_KEY="$SECRET_KEY" \
    -e ENABLE_OLLAMA_API=false -e ENABLE_OPENAI_API=false \
    -e VERTEX_AI_RAG_STORE="$VERTEX_RAG_STORE" \
    -e PYTHONUNBUFFERED=1 -e SEND_TO_LOG_ANALYTICS=false \
    -e RAG_EMBEDDING_ENGINE=openai -e PYTHONWARNINGS=always::ResourceWarning \
    "$IMAGE" bash -c 'mkdir -p /tmp/e2e; bash start.sh 2>&1 | tee -a /tmp/e2e/server.log' \
    >/dev/null || die "docker run failed (port ${PORT:-auto} busy? image $IMAGE missing?)"
}

wait_healthy() {
  local t0 now
  t0=$(date +%s)
  while true; do
    [ "$(dk inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ] \
      || die "container $NAME stopped, see $OUT/server.log"
    if dk exec "$NAME" python3 -c 'import urllib.request as u; u.urlopen("http://127.0.0.1:8080/health", timeout=3)' \
      >/dev/null 2>&1; then
      echo "Open WebUI healthy after $(( $(date +%s) - t0 ))s (http://localhost:$(host_port))"
      return 0
    fi
    now=$(date +%s)
    [ $((now - t0)) -lt "$HEALTH_TIMEOUT" ] || die "Open WebUI not healthy after ${HEALTH_TIMEOUT}s"
    sleep 2
  done
}

# Called as `copy_into_container || die ...`, which turns set -e off inside.
copy_into_container() {
  dk exec "$NAME" bash -c 'rm -rf /e2e && mkdir -p /e2e/out' || return 1
  tar -C "$HERE" -cf - --exclude=out --exclude=__pycache__ \
      e2e.py harness suites mocks probe \
    | dk exec -i "$NAME" tar -C /e2e -xf - || return 1
  tar -C "$OUT" -cf - functions | dk exec -i "$NAME" tar -C /e2e -xf -
}

start_mocks() {
  if [ "$REUSE" = 1 ]; then
    dk exec "$NAME" python3 /e2e/mocks/serve_all.py --shutdown >/dev/null 2>&1 || true
    sleep 1
  fi
  # E2E_MOCK_FAULT=500: every provider route of the mocks answers HTTP 500
  dk exec -d -e E2E_MOCK_FAULT="${E2E_MOCK_FAULT:-}" -w /e2e/mocks "$NAME" \
    bash -c 'exec python3 -u serve_all.py >>/e2e/out/mocks.txt 2>&1'
}

# ---------------------------------------------------------------- main
trap cleanup EXIT
trap 'exit 130' INT TERM
T0=$(date +%s)
stage_functions
if [ "$REUSE" = 1 ]; then
  dk container inspect "$NAME" >/dev/null 2>&1 || die "--reuse: no container $NAME"
  echo "reusing $NAME"
else
  start_container
fi
wait_healthy
copy_into_container || die "copying the test files into $NAME failed"
start_mocks || die "starting the mocks in $NAME failed"

set +e
dk exec -e E2E_IMAGE="$IMAGE" "$NAME" \
  python3 -u /e2e/e2e.py --suites "$SUITES" --out /e2e/out ${ONLY:+--only "$ONLY"} $VERBOSE \
  ${STRICT:+--strict-known} 2>&1 | tee "$OUT/driver.txt"
RC=${PIPESTATUS[0]}
set -e
case "$RC" in
  0 | 1 | 2) ;;
  *) echo "error: the driver exited with $RC (docker exec failed?)" >&2; RC=2 ;;
esac
dk exec "$NAME" tar -C /e2e/out -cf - . | tar -C "$OUT" -xf - || true
echo "total runtime: $(( $(date +%s) - T0 ))s"
exit "$RC"
