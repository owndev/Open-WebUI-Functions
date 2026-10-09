#!/usr/bin/env bash
# Real-API smoke test of the Gemini pipe's native tool calling: one scenario per
# risk R1-R15 of the 1.18.0 spec, the server-side behaviour the mock-based
# harness cannot confirm. Manual and opt-in: it needs a Gemini API key and
# makes billed requests (about 40 small ones).
#
#   GOOGLE_API_KEY=... tests/e2e/smoke/gemini-tools.sh [--image v0.11.4-slim] \
#       [--model gemini-3-flash-preview] [--model-25 gemini-2.5-flash]
#   tests/e2e/smoke/gemini-tools.sh --dry-run-mock   # no key: against the e2e mock
#
# Starts a fresh Open WebUI container with run.sh's helpers, installs
# google_gemini.py and the google_search_tool filter from this worktree, the tool
# mocks and the e2e workspace tool, and runs smoke/gemini_tools.py inside the
# container. The pipe talks to generativelanguage.googleapis.com through a
# recording pass-through proxy (smoke/proxy.py); --dry-run-mock forwards to the
# e2e Gemini mock instead. Container and volume are removed on exit (also after
# Ctrl-C); images are never removed. The key is read from the environment only,
# stored in the encrypted valve, never printed, and every output file is scanned
# for it and for the secret values of the Vertex credentials file (a hit is
# redacted and fails the run). The proxy forwards at most --max-requests generate
# requests and answers any further one with HTTP 429 itself.
#
# Docs: docs/testing.md, "Real-API smoke test (manual, needs a key)"
set -euo pipefail

SMOKE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Sourced, run.sh only defines its helpers (dk, start_container, wait_healthy,
# start_mocks, to_lf, wait_for_exit, PACK_OUTPUT, ...) and HERE / REPO.
# shellcheck source=../run.sh
source "$SMOKE_DIR/../run.sh"
unset E2E_MOCK_FAULT  # the tool mocks must answer, the dry run's Gemini mock too

SMOKE_FILES=(pipelines/google/google_gemini.py filters/google_search_tool.py)
RUNNER=/e2e/smoke/gemini_tools.py
DRY_KEY=mock-smoke-key-not-a-secret

smoke_usage() {
  cat <<EOF
usage: GOOGLE_API_KEY=... tests/e2e/smoke/gemini-tools.sh [options]
       tests/e2e/smoke/gemini-tools.sh --dry-run-mock [options]

Real-API smoke test of google_gemini.py's native tool calling (spec risks
R1-R15). The key comes from the environment only, never from an argument.

options:
  -i, --image IMAGE        Open WebUI image or tag (default: \$OWUI_IMAGE or
                           $DEFAULT_IMAGE)
      --model ID           Gemini 3 model (default gemini-3-flash-preview)
      --model-25 ID        Gemini 2.5 model (default gemini-2.5-flash)
      --model-latest ID    alias tried in R4 when the API lists it
                           (default gemini-flash-latest; '' skips it)
      --only LIST          comma separated risks, e.g. R2,R5 (default: all)
      --retries N          extra attempts of a scenario whose model did not do
                           what the prompt asked or hit a temporary error
                           (default 2)
      --turns N            tool turns of R13 (default 4)
      --max-tokens N       max_tokens of every request (default 2048)
      --max-requests N     hard cap of generate requests sent upstream; the
                           proxy answers any further one with HTTP 429 without
                           forwarding it (default: 2x the estimated maximum the
                           runner prints)
      --api-version V      API_VERSION valve (default: the pipe's default)
      --dry-run-mock       no key: the proxy forwards to the e2e Gemini mock
                           (checks the runner; results differ from the real API)
      --src DIR            the function files from DIR (another checkout)
      --vertex-credentials FILE  service account or ADC JSON for R9 (default
                           \$GOOGLE_APPLICATION_CREDENTIALS); without it R9 is SKIP
      --vertex-project ID  (default \$GOOGLE_CLOUD_PROJECT, else the file's
                           project_id)
      --vertex-location L  (default global)
      --vertex-rag-store S Vertex AI Search data store tried in R9 (optional)
  -n, --name NAME          container name (default owui-smoke-<time>-<random>)
  -p, --port PORT          host port of the UI on 127.0.0.1 (default: free port)
  -o, --out DIR            output directory (default tests/e2e/out/<time>-<name>)
  -k, --keep               keep container + volume (they hold the key, encrypted,
                           and with --vertex-credentials the credentials file
                           in plain text)
      --force              delete a volume NAME-data left over from a kept run
  -h, --help               this help

environment: GOOGLE_API_KEY (required without --dry-run-mock), SMOKE_TIMEOUT
(seconds, default 2400), E2E_HEALTH_TIMEOUT (default 600), E2E_SECRET_KEY

exit code: 0 = no FAIL, 1 = at least one FAIL (or the key found in an output
file), 2 = setup error (install, preflight: key, model, network)
EOF
}

# ---------------------------------------------------------------- arguments
smoke_parse_args() {
  IMAGE=${OWUI_IMAGE:-$DEFAULT_IMAGE}
  NAME=""
  PORT=""
  OUT=""
  KEEP=0
  FORCE=0
  REUSE=0
  DRY=0
  SRC_DIR=""
  RUNNER_ARGS=()
  VERTEX_CRED=${GOOGLE_APPLICATION_CREDENTIALS:-}
  VERTEX_PROJECT=${GOOGLE_CLOUD_PROJECT:-}
  VERTEX_RAG=""
  HEALTH_TIMEOUT=${E2E_HEALTH_TIMEOUT:-600}
  SMOKE_TIMEOUT=${SMOKE_TIMEOUT:-2400}
  SECRET_KEY=${E2E_SECRET_KEY:-e2e-local-secret-key-0123456789abcdef}
  # run.sh starts the container with a fake VERTEX_AI_RAG_STORE (the pipe reads
  # the variable): empty here, R9 sets the data store through the valve.
  VERTEX_RAG_STORE=""
  while [ $# -gt 0 ]; do
    case "$1" in
      -i|--image) need "$@"; IMAGE=$2; shift 2 ;;
      --only)
        need "$@"
        local risk
        for risk in ${2//,/ }; do
          case "$risk" in
            [Rr][1-9] | [Rr]1[0-5]) ;;
            *) die "--only: unknown risk '$risk' (R1 ... R15)" ;;
          esac
        done
        RUNNER_ARGS+=("$1" "$2"); shift 2 ;;
      --model|--model-25|--retries|--turns|--max-tokens|--api-version|--vertex-location)
        need "$@"; RUNNER_ARGS+=("$1" "$2"); shift 2 ;;
      --max-requests)
        need "$@"
        is_seconds "$2" || die "--max-requests needs a number > 0, got '$2'"
        RUNNER_ARGS+=("$1" "$2"); shift 2 ;;
      --model-latest)
        [ $# -ge 2 ] || die "option $1 needs a value (see --help)"
        RUNNER_ARGS+=("$1=$2"); shift 2 ;;
      --dry-run-mock) DRY=1; shift ;;
      --src) need "$@"; SRC_DIR=$2; shift 2 ;;
      --vertex-credentials) need "$@"; VERTEX_CRED=$2; shift 2 ;;
      --vertex-project) need "$@"; VERTEX_PROJECT=$2; shift 2 ;;
      --vertex-rag-store) need "$@"; VERTEX_RAG=$2; shift 2 ;;
      -n|--name) need "$@"; NAME=$2; shift 2 ;;
      -p|--port) need "$@"; PORT=$2; shift 2 ;;
      -o|--out) need "$@"; OUT=$2; shift 2 ;;
      -k|--keep) KEEP=1; shift ;;
      --force) FORCE=1; shift ;;
      -h|--help) smoke_usage; exit 0 ;;
      *) smoke_usage >&2; die "unknown argument $1" ;;
    esac
  done
}

smoke_validate() {
  command -v docker >/dev/null || die "docker not found"
  case "$IMAGE" in */*) ;; *) IMAGE="ghcr.io/open-webui/open-webui:$IMAGE" ;; esac
  is_seconds "$SMOKE_TIMEOUT" || die "SMOKE_TIMEOUT must be a number of seconds > 0"
  is_seconds "$HEALTH_TIMEOUT" || die "E2E_HEALTH_TIMEOUT must be a number of seconds > 0"
  [ -z "$SRC_DIR" ] || [ -d "$SRC_DIR" ] || die "--src $SRC_DIR is not a directory"
  if [ "$DRY" = 1 ]; then
    SMOKE_KEY=$DRY_KEY
    if [ -n "$VERTEX_CRED" ]; then
      echo "note: --dry-run-mock does not use Vertex credentials (R9 is SKIP)"
      VERTEX_CRED=""
    fi
  else
    SMOKE_KEY=${GOOGLE_API_KEY:-}
    [ "${#SMOKE_KEY}" -ge 6 ] || die "GOOGLE_API_KEY is not set: export it (never" \
      "pass it as an argument), or check the runner with --dry-run-mock"
  fi
  if [ -n "$VERTEX_CRED" ]; then
    [ -f "$VERTEX_CRED" ] || die "Vertex credentials $VERTEX_CRED not found"
    local secret
    while IFS= read -r secret; do
      VERTEX_SECRETS+=("$secret")
    done < <(smoke_vertex_secrets "$VERTEX_CRED")
  fi
  NAME=${NAME:-owui-smoke-$(date +%H%M%S)-$RANDOM}
  VOLUME="$NAME-data"
  OUT=${OUT:-$HERE/out/$(date +%Y%m%d-%H%M%S)-$NAME}
  mkdir -p "$OUT"
  OUT=$(cd "$OUT" && pwd)
}

# Secret values of a Vertex credentials file, one per line, as the runner's
# vertex_secrets() takes them: refresh_token, client_secret, private_key_id and
# the base64 lines of the PEM private_key (JSON-escaped in the file and in a log
# line); values under 16 characters and the BEGIN / END lines are left out. No
# JSON parser on the host: these values never contain a quote.
smoke_vertex_secrets() {
  local keys='"(private_key|private_key_id|client_secret|refresh_token)"'
  grep -oE "$keys"'[[:space:]]*:[[:space:]]*"[^"]*"' "$1" \
    | sed -E 's/^"[a-z_]+"[[:space:]]*:[[:space:]]*"//; s/"$//' \
    | awk '{ gsub(/\\r/, ""); gsub(/\\n/, "\n"); gsub(/\\\//, "/"); print }' \
    | awk 'length($0) >= 16 && $0 !~ /^-----/' || true
}

# ------------------------------------------------------------------- files
smoke_stage() {
  local staged="$OUT/functions" path src
  rm -rf "$staged"
  mkdir -p "$staged"
  : >"$staged/SOURCES.txt"
  for path in "${SMOKE_FILES[@]}"; do
    mkdir -p "$staged/$(dirname "$path")"
    if [ -n "$SRC_DIR" ]; then
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

# Called as `smoke_copy || die ...`, which turns set -e off inside.
smoke_copy() {
  dk exec "$NAME" bash -c 'rm -rf /e2e && mkdir -p /e2e/out' || return 1
  tar -C "$HERE" -cf - --exclude=out --exclude=__pycache__ \
      harness suites mocks probe smoke \
    | dk exec -i "$NAME" tar -C /e2e -xf - || return 1
  tar -C "$OUT" -cf - functions | dk exec -i "$NAME" tar -C /e2e -xf -
}

# Vertex AI: the credentials become the container's Application Default
# Credentials file (google-auth finds it there; the Open WebUI process has no
# GOOGLE_APPLICATION_CREDENTIALS). The file is streamed through stdin, never
# copied into the output directory.
smoke_vertex() {
  local home adc
  home=$(dk exec "$NAME" sh -c 'printf %s "$HOME"') || return 1
  adc="${home:-/root}/.config/gcloud/application_default_credentials.json"
  VERTEX_ADC=$adc
  # shellcheck disable=SC2016 # $1 belongs to the sh -c script
  dk exec -i "$NAME" sh -c 'mkdir -p "${1%/*}" && umask 077 && cat >"$1"' sh "$adc" \
    <"$VERTEX_CRED" || return 1
  RUNNER_ARGS+=(--vertex-adc "$adc")
  [ -z "$VERTEX_PROJECT" ] || RUNNER_ARGS+=(--vertex-project "$VERTEX_PROJECT")
  [ -z "$VERTEX_RAG" ] || RUNNER_ARGS+=(--vertex-rag-store "$VERTEX_RAG")
  echo "Vertex AI: credentials installed as the container's ADC file (R9 runs)"
}

# ------------------------------------------------------------------ runner
RUNNER_PID=""
RUNNER_STARTED=0
FETCHED=0
VERTEX_ADC=""  # set once the credentials file is copied into the container
VERTEX_SECRETS=()

# Runs the runner; sets RC. In the background, so Ctrl-C reaches the traps at once.
smoke_run() {
  RUNNER_STARTED=1
  local extra=()
  [ "$DRY" = 0 ] || extra+=(--dry-run-mock)
  # docker reads GOOGLE_API_KEY from its environment (-e NAME without a value):
  # the key is never part of a command line
  GOOGLE_API_KEY="$SMOKE_KEY" dk exec -e GOOGLE_API_KEY -e PYTHONPATH=/e2e \
    -e E2E_IMAGE="$IMAGE" -w /e2e "$NAME" \
    timeout -k 30 "$((SMOKE_TIMEOUT + 60))" \
    python3 -u "$RUNNER" --out /e2e/out --timeout "$SMOKE_TIMEOUT" \
    "${extra[@]+"${extra[@]}"}" "${RUNNER_ARGS[@]+"${RUNNER_ARGS[@]}"}" 2>&1 \
    | tee "$OUT/driver.txt" &
  RUNNER_PID=$!
  RC=0
  wait "$RUNNER_PID" || RC=$?
  RUNNER_PID=""
  case "$RC" in
    0 | 1 | 2) ;;
    *) echo "error: the runner exited with $RC" >&2; RC=2 ;;
  esac
}

# SIGINT to the runner in the container (it writes partial results), SIGKILL
# when it still runs 15 s later. The image has no ps / pkill: /proc is scanned.
smoke_stop_runner() {
  dk exec -i "$NAME" python3 - >/dev/null 2>&1 <<'PY'
import os
import signal
import time


def runners():
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
        if program.startswith(b"python") and b"/e2e/smoke/gemini_tools.py" in argv:
            found.add(int(pid))
    return found


pids = runners()
for pid in pids:
    os.kill(pid, signal.SIGINT)
deadline = time.monotonic() + 15
while pids & runners() and time.monotonic() < deadline:
    time.sleep(0.3)
for pid in pids & runners():
    os.kill(pid, signal.SIGKILL)
PY
}

smoke_fetch() {
  FETCHED=1
  dk exec "$NAME" bash -c "$PACK_OUTPUT" | tar -C "$OUT" -xf - || true
}

# Every output file is scanned for the key and the Vertex secret values (also
# the server log and the mocks' output); a hit is redacted in place and fails
# the run.
smoke_scan_secrets() {
  local leaked=() secrets=() file secret what="the key"
  [ "${#SMOKE_KEY}" -lt 6 ] || secrets+=("$SMOKE_KEY")
  for secret in "${VERTEX_SECRETS[@]+"${VERTEX_SECRETS[@]}"}"; do
    [ "${#secret}" -lt 6 ] || secrets+=("$secret")
  done
  [ ${#secrets[@]} -gt 0 ] || return 0
  [ ${#VERTEX_SECRETS[@]} -eq 0 ] ||
    what="the key and the ${#VERTEX_SECRETS[@]} Vertex secret values"
  while IFS= read -r file; do
    [ -n "$file" ] || continue
    leaked+=("${file#"$OUT"/}")
    SMOKE_SECRETS=$(printf '%s\n' "${secrets[@]}") awk '
      BEGIN { n = split(ENVIRON["SMOKE_SECRETS"], s, "\n") }
      {
        line = $0
        for (k = 1; k <= n; k++) {
          if (s[k] == "") continue
          out = ""
          while ((i = index(line, s[k])) > 0) {
            out = out substr(line, 1, i - 1) "***"
            line = substr(line, i + length(s[k]))
          }
          line = out line
        }
        print line
      }' "$file" >"$file.redacted" && mv "$file.redacted" "$file"
  done < <(printf '%s\n' "${secrets[@]}" | grep -rlF -f - "$OUT" 2>/dev/null)
  if [ ${#leaked[@]} -gt 0 ]; then
    echo "FAIL secret scan: $what: found in ${leaked[*]} (redacted there now)" >&2
    return 1
  fi
  echo "secret scan: $what: in none of the $(find "$OUT" -type f | wc -l)" \
    "output files"
}

# Runs on every exit (also after Ctrl-C / SIGTERM).
smoke_cleanup() {
  local rc=$?
  set +e
  trap '' INT TERM HUP
  if [ "$CREATED" = 1 ]; then
    if [ -n "$RUNNER_PID" ]; then
      echo "interrupted: stopping the runner in $NAME (partial results)" >&2
      smoke_stop_runner
      wait_for_exit "$RUNNER_PID" 10
    fi
    if [ "$RUNNER_STARTED" = 1 ] && [ "$FETCHED" = 0 ]; then smoke_fetch; fi
    [ -s "$OUT/server.log" ] || dk logs "$NAME" >"$OUT/server.log" 2>&1
    if [ "$KEEP" = 1 ]; then
      echo "kept container $NAME (volume $VOLUME; it holds the key, encrypted):" \
        "http://localhost:$(host_port) login admin@example.com / Passw0rd!e2e"
      [ -z "$VERTEX_ADC" ] ||
        echo "  it also holds the Vertex credentials file in plain text:" \
          "$VERTEX_ADC (removed with the container)"
      echo "  cleanup: docker rm -f $NAME && docker volume rm $VOLUME"
    else
      dk rm -f "$NAME" >/dev/null
      dk volume rm "$VOLUME" >/dev/null
    fi
  fi
  if ! smoke_scan_secrets && [ "$rc" = 0 ]; then rc=1; fi
  [ ! -f "$OUT/summary.md" ] || echo "summary: $OUT/summary.md"
  echo "output: $OUT"
  exit "$rc"
}

# --------------------------------------------------------------------- main
smoke_main() {
  smoke_parse_args "$@"
  smoke_validate
  trap smoke_cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  local t0
  t0=$(date +%s)
  if [ "$DRY" = 1 ]; then
    echo "DRY RUN: the pipe talks to the e2e Gemini mock, no key needed"
  else
    echo "REAL API: requests go to generativelanguage.googleapis.com and are" \
      "billed to the key's project"
  fi
  smoke_stage
  start_container
  wait_healthy
  smoke_copy || die "copying the test files into $NAME failed"
  if [ -n "$VERTEX_CRED" ]; then
    smoke_vertex || die "copying the Vertex credentials into $NAME failed"
  fi
  start_mocks || die "starting the mocks in $NAME failed"
  smoke_run
  smoke_fetch
  echo "total runtime: $(( $(date +%s) - t0 ))s"
  exit "$RC"
}

smoke_main "$@"
