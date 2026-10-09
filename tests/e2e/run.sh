#!/usr/bin/env bash
# Local end-to-end tests for the Open WebUI functions in this repo.
#
# Starts a throw-away Open WebUI container, copies the function files, the
# provider mocks and the driver into it, runs the scenario suites INSIDE the
# container and removes container + volume again. Host needs: Docker and bash
# (Linux, macOS, Git Bash on Windows). No host Python.
#
# The whole script runs inside main() (called at the end), so editing or
# checking out another version of this file during a run does not break it.
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

# Default Open WebUI image (.github/workflows/e2e.yml reads this line).
DEFAULT_IMAGE=ghcr.io/open-webui/open-webui:v0.11.4-slim

FUNCTION_FILES=(
  pipelines/google/google_gemini.py
  pipelines/azure/azure_ai_foundry.py
  pipelines/n8n/n8n.py
  pipelines/infomaniak/infomaniak.py
  filters/google_search_tool.py
  filters/vertex_ai_search_tool.py
  filters/time_token_tracker.py
)

# The driver gets E2E_TIMEOUT seconds; `timeout` stops it KILL_GRACE seconds
# later (SIGTERM: it writes its partial results) and kills it KILL_GRACE
# seconds after that.
KILL_GRACE=60

# Suites are the modules in tests/e2e/suites/ (no registration needed).
available_suites() {
  local f
  for f in "$HERE"/suites/*.py; do
    f=${f##*/}
    f=${f%.py}
    case "$f" in _*) ;; *) printf '%s ' "$f" ;; esac
  done
}

usage() {
  echo "usage: tests/e2e/run.sh [options] [suite ...]"
  echo
  echo "suites: $(available_suites)all (default: all)"
  cat <<EOF

options:
  -i, --image IMAGE    Open WebUI image or tag (default: \$OWUI_IMAGE or
                       $DEFAULT_IMAGE); a bare tag
                       like v0.11.3-slim means ghcr.io/open-webui/open-webui:TAG
  -s, --suites LIST    comma separated suites (same as positional suites)
      --only REGEX     run only scenario groups matching REGEX on "<suite>.<group>"
                       (Python regex; alternatives as 'gemini.(api|image)')
      --ref REF        test the function files as of git REF (git show REF:path)
      --src DIR        test the function files from DIR (another checkout/worktree)
      --file PATH=FILE use FILE for repo path PATH (repeatable), e.g.
                       --file pipelines/azure/azure_ai_foundry.py=/tmp/fix.py
                       (PATH must be one of the function files listed below)
  -n, --name NAME      container name (default: \$E2E_NAME or owui-e2e-<random>);
                       the volume is NAME-data
  -p, --port PORT      host port for the UI on 127.0.0.1 (default: \$E2E_PORT or
                       a free port chosen by Docker)
  -o, --out DIR        output directory (default: \$E2E_OUT or
                       tests/e2e/out/<timestamp>-<name>)
  -k, --keep           keep container + volume running afterwards (debugging;
                       default: \$E2E_KEEP=1)
      --reuse          reuse the running container NAME (implies --keep): skips
                       start-up, stops drivers left over from an interrupted run,
                       re-copies files, restarts the mocks
      --force          delete a volume NAME-data left over from an earlier run
                       (without it run.sh refuses to start)
  -v, --verbose        print details of passing scenarios too
      --strict-known   a check tagged with a known bug that passes while its
                       marker still applies is a FAIL (default: \$E2E_STRICT_KNOWN)
  -h, --help           this help

environment:
  E2E_TIMEOUT=1800       time budget of the driver in seconds: a suite still
                         running when it is used up is a <suite>.timeout FAIL
                         (the driver is stopped ${KILL_GRACE}s later)
  E2E_SUITE_TIMEOUT=900  time limit of one suite in seconds
  E2E_HEALTH_TIMEOUT=600 seconds to wait for Open WebUI to become healthy
  E2E_SECRET_KEY         WEBUI_SECRET_KEY of the container (default: a fixed key)
  E2E_MOCK_FAULT=500     every provider route of the mocks answers this status

exit code: 0 = only PASS/KNOWN, 1 = at least one FAIL, 2 = setup error
(130 / 143 after Ctrl-C / SIGTERM; partial results are kept)
EOF
  echo
  echo "function files: ${FUNCTION_FILES[*]}"
}

die() { echo "error: $*" >&2; exit 2; }
need() {
  [ $# -ge 2 ] || die "option $1 needs a value (see --help)"
  case "$2" in -?*) die "option $1 needs a value, got the option '$2' (see --help)" ;; esac
}
add_suites() { SUITES="${SUITES:+$SUITES,}$1"; }
# repo path as given to --file: backslashes -> slashes, no leading ./
norm_path() { local p=${1//\\//}; echo "${p#./}"; }
# a whole number of seconds > 0
is_seconds() { case "$1" in '' | *[!0-9]* | 0*) return 1 ;; esac; }

# ---------------------------------------------------------------- arguments
parse_args() {
  IMAGE=${OWUI_IMAGE:-$DEFAULT_IMAGE}
  IMAGE_GIVEN=${OWUI_IMAGE:+1}
  NAME=${E2E_NAME:-}
  PORT=${E2E_PORT:-}
  OUT=${E2E_OUT:-}
  KEEP=${E2E_KEEP:-0}
  REUSE=0
  FORCE=0
  ONLY=""
  REF=""
  SRC_DIR=""
  VERBOSE=""
  STRICT=${E2E_STRICT_KNOWN:-}
  SUITES=""
  HEALTH_TIMEOUT=${E2E_HEALTH_TIMEOUT:-600}
  RUN_TIMEOUT=${E2E_TIMEOUT:-1800}
  SUITE_TIMEOUT=${E2E_SUITE_TIMEOUT:-}
  SECRET_KEY=${E2E_SECRET_KEY:-e2e-local-secret-key-0123456789abcdef}
  VERTEX_RAG_STORE="projects/e2e/locations/global/collections/default_collection/dataStores/e2e-store"
  OVERRIDES=()

  while [ $# -gt 0 ]; do
    case "$1" in
      -i|--image) need "$@"; IMAGE=$2; IMAGE_GIVEN=1; shift 2 ;;
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
      --force) FORCE=1; shift ;;
      -v|--verbose) VERBOSE=--verbose; shift ;;
      --strict-known) STRICT=1; shift ;;
      -h|--help) usage; exit 0 ;;
      -*) usage >&2; die "unknown option $1" ;;
      *) add_suites "$1"; shift ;;
    esac
  done
}

validate() {
  local suite override wanted match path
  case "$STRICT" in 1 | true | yes | on) STRICT=1 ;; *) STRICT="" ;; esac
  case "$IMAGE" in */*) ;; *) IMAGE="ghcr.io/open-webui/open-webui:$IMAGE" ;; esac
  SUITES=${SUITES:-all}
  case ",$SUITES," in *,all,*) SUITES=all ;; esac
  for suite in ${SUITES//,/ }; do
    case "$suite" in
      all) ;;
      _* | *[!a-z0-9_]*) die "unknown suite '$suite' (choose from: $(available_suites)all)" ;;
      *) [ -f "$HERE/suites/$suite.py" ] \
           || die "unknown suite '$suite' (choose from: $(available_suites)all)" ;;
    esac
  done
  is_seconds "$RUN_TIMEOUT" || die "E2E_TIMEOUT must be a number of seconds > 0, got '$RUN_TIMEOUT'"
  [ -z "$SUITE_TIMEOUT" ] || is_seconds "$SUITE_TIMEOUT" \
    || die "E2E_SUITE_TIMEOUT must be a number of seconds > 0, got '$SUITE_TIMEOUT'"
  is_seconds "$HEALTH_TIMEOUT" \
    || die "E2E_HEALTH_TIMEOUT must be a number of seconds > 0, got '$HEALTH_TIMEOUT'"
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
}

# ---------------------------------------------------------------- stage files
# Windows checkouts (core.autocrlf) have CRLF line endings, --ref stages the
# blob as stored (some files are stored with CRLF) and users paste LF: every
# staged file is converted to LF. Returns 0 when the file had CRLF line endings.
# -U: Git for Windows' grep strips the CR of CRLF lines unless it reads the file
# as binary, so a plain grep never finds one there.
to_lf() {
  grep -qU $'\r' "$1" || return 1
  awk '{ sub(/\r$/, ""); print }' "$1" >"$1.lf" && mv "$1.lf" "$1"
}

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
    if to_lf "$staged/$path"; then src="$src, CRLF converted to LF"; fi
    printf '%s\t%s\n' "$path" "$src" >>"$staged/SOURCES.txt"
  done
}

# ---------------------------------------------------------------- container
CREATED=0
DRIVER_PID=""     # docker exec | tee pipeline of the running driver
DRIVER_STARTED=0
FETCHED=0

# Runs on every exit (also after Ctrl-C / SIGTERM): stops a running driver so it
# writes its partial results, copies the output and removes container + volume
# (unless kept).
cleanup() {
  local rc=$?
  set +e
  trap '' INT TERM HUP  # a second Ctrl-C must not cut the cleanup short
  # Every docker call takes a second or more on a busy Docker Desktop, so after a
  # driver run (the container exists) the inspect is skipped, and stopping the
  # driver, copying the output and reading the server log is one call.
  if [ "$DRIVER_STARTED" = 1 ] || dk container inspect "$NAME" >/dev/null 2>&1; then
    if [ -n "$DRIVER_PID" ]; then
      echo "interrupted: stopping the driver in $NAME" >&2
      FETCHED=1
      stop_drivers 5 fetch | tar -C "$OUT" -xf - 2>/dev/null
      wait_for_exit "$DRIVER_PID" 5  # tee writes the driver's last lines
      DRIVER_PID=""
    elif [ "$DRIVER_STARTED" = 1 ] && [ "$FETCHED" = 0 ]; then
      fetch_output
    fi
    # no copied server log (the driver never ran, the copy failed): ask Docker
    [ -s "$OUT/server.log" ] || dk logs "$NAME" >"$OUT/server.log" 2>&1
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

# wait_for_exit PID SECONDS: wait until a background process ended (at most SECONDS)
wait_for_exit() {
  local i
  for ((i = 0; i < $2 * 5; i++)); do
    kill -0 "$1" 2>/dev/null || return 0
    sleep 0.2
  done
  return 1
}

host_port() { dk port "$NAME" 8080/tcp 2>/dev/null | head -n1 | sed 's/.*://'; }

start_container() {
  if dk container inspect "$NAME" >/dev/null 2>&1; then
    die "container $NAME already exists (use --reuse, or remove it)"
  fi
  if dk volume inspect "$VOLUME" >/dev/null 2>&1; then
    [ "$FORCE" = 1 ] || die "volume $VOLUME already exists (left over from a kept run?);" \
      "remove it with 'docker volume rm $VOLUME' or pass --force"
    dk volume rm "$VOLUME" >/dev/null || die "cannot remove volume $VOLUME (in use?)"
  fi
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

# --reuse: the container must exist; the image it runs is what results.json reports.
reuse_container() {
  local actual stopped
  dk container inspect "$NAME" >/dev/null 2>&1 || die "--reuse: no container $NAME"
  actual=$(dk inspect -f '{{.Config.Image}}' "$NAME")
  if [ -n "$IMAGE_GIVEN" ] && [ "$actual" != "$IMAGE" ]; then
    echo "note: --reuse runs the image of $NAME ($actual), not $IMAGE" >&2
  fi
  IMAGE=$actual
  echo "reusing $NAME ($IMAGE)"
  [ "$(dk inspect -f '{{.State.Running}}' "$NAME")" = true ] \
    || die "--reuse: container $NAME is not running"
  # A driver of an earlier run that was killed (Ctrl-C on an old run.sh, closed
  # terminal, SIGKILL) keeps running inside the container and would share the
  # mocks and the server log with this run.
  stopped=$(stop_drivers 5) || die "--reuse: cannot list the processes in $NAME"
  case "$stopped" in
    "0 "*) ;;
    *) echo "stopped ${stopped%% *} driver(s) left over from an earlier run in $NAME" ;;
  esac
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

# Container side of copying the output: /e2e/out plus the server log (the same
# text as `docker logs`, which would cost another docker call) as a tar stream.
PACK_OUTPUT='cp /tmp/e2e/server.log /e2e/out/server.log 2>/dev/null; exec tar -C /e2e/out -cf - .'

# stop_drivers SECONDS [fetch]: stops the e2e.py driver processes in the
# container (the image has no ps/pkill, so /proc is scanned): SIGINT first, so a
# driver writes its partial results, then SIGKILL for those still running after
# SECONDS. Prints "<found> <killed>"; with "fetch" it then prints the output as
# PACK_OUTPUT does instead (one docker call on Ctrl-C instead of two).
stop_drivers() {
  # shellcheck disable=SC2016 # $0 / $1 belong to the bash -c script
  dk exec -i "$NAME" bash -c \
    'counts=$(python3 - "$0") || exit; [ -z "$1" ] || exec bash -c "$1"; echo "$counts"' \
    "$1" "${2:+$PACK_OUTPUT}" <<'PY'
import os
import signal
import sys
import time


def drivers():
    found = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                argv = fh.read().split(b"\0")
        except OSError:
            continue
        program = argv[0].rsplit(b"/", 1)[-1]
        if program.startswith(b"python") and b"/e2e/e2e.py" in argv:
            found.add(int(pid))
    return found


pids = drivers()
for pid in pids:
    try:
        os.kill(pid, signal.SIGINT)
    except OSError:
        pass
deadline = time.monotonic() + float(sys.argv[1])
while pids & drivers() and time.monotonic() < deadline:
    time.sleep(0.2)
left = pids & drivers()
for pid in left:
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
print(len(pids), len(left))
PY
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

# Runs the driver; sets RC. The pipeline runs in the background and the script
# waits for it, so the INT / TERM traps (and with them cleanup) fire at once
# instead of after the driver ended.
run_driver() {
  DRIVER_STARTED=1
  dk exec -e E2E_IMAGE="$IMAGE" -e E2E_TIMEOUT="$RUN_TIMEOUT" \
    ${SUITE_TIMEOUT:+-e "E2E_SUITE_TIMEOUT=$SUITE_TIMEOUT"} "$NAME" \
    timeout -k "$KILL_GRACE" "$((RUN_TIMEOUT + KILL_GRACE))" \
    python3 -u /e2e/e2e.py --suites "$SUITES" --out /e2e/out ${ONLY:+--only "$ONLY"} \
    $VERBOSE ${STRICT:+--strict-known} 2>&1 | tee "$OUT/driver.txt" &
  DRIVER_PID=$!
  RC=0
  wait "$DRIVER_PID" || RC=$?
  DRIVER_PID=""
  case "$RC" in
    0 | 1 | 2) ;;
    124 | 137)
      echo "error: the driver was still running $((RUN_TIMEOUT + KILL_GRACE))s after" \
        "its start (E2E_TIMEOUT=$RUN_TIMEOUT) and was stopped" >&2
      RC=2
      ;;
    *) echo "error: the driver exited with $RC (docker exec failed?)" >&2; RC=2 ;;
  esac
}

fetch_output() {
  FETCHED=1
  dk exec "$NAME" bash -c "$PACK_OUTPUT" | tar -C "$OUT" -xf - || true
}

# ---------------------------------------------------------------- main
main() {
  parse_args "$@"
  validate
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  T0=$(date +%s)
  rm -f "$OUT/server.log"  # cleanup only asks Docker when no fresh copy arrived
  stage_functions
  if [ "$REUSE" = 1 ]; then
    reuse_container
  else
    start_container
  fi
  wait_healthy
  copy_into_container || die "copying the test files into $NAME failed"
  start_mocks || die "starting the mocks in $NAME failed"
  run_driver
  fetch_output
  echo "total runtime: $(( $(date +%s) - T0 ))s"
  exit "$RC"
}

# main never returns (every path ends in exit or die). Sourced, the file only
# defines its helpers (tests/e2e/smoke/gemini-tools.sh reuses the container ones).
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
