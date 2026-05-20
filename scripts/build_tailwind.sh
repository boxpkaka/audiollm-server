#!/usr/bin/env bash
# Build the precompiled Tailwind stylesheet for the demo frontend.
#
# Why this script exists:
#   The HTML pages used to load Tailwind from cdn.tailwindcss.com, which
#   ships a JIT runtime that re-scans the page on every navigation. That
#   was the largest single source of perceived "stutter" when switching
#   between the ASR / TS-ASR / Emotion pages. We instead pre-compile a
#   single ``frontend/tailwind.css`` and check it into the repo, so the
#   server can hand the browser a static stylesheet that's instantly
#   parseable and fully cacheable.
#
# Usage:
#   bash scripts/build_tailwind.sh
#   bash scripts/build_tailwind.sh --watch    # rebuild on file changes
#
# Re-run whenever a Tailwind utility class is added/removed in
# ``frontend/*.html`` or any of the JS files that emit markup
# (``frontend/sidebar.js``, ``frontend/emotion-app.js``, ...).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FRONTEND_DIR="${REPO_ROOT}/frontend"
INPUT_CSS="${FRONTEND_DIR}/tailwind.input.css"
OUTPUT_CSS="${FRONTEND_DIR}/tailwind.css"
CONFIG_FILE="${FRONTEND_DIR}/tailwind.config.js"

EXTRA_ARGS=()
WATCH=0
for arg in "$@"; do
  case "$arg" in
    -w|--watch) WATCH=1 ;;
    -h|--help)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    *) EXTRA_ARGS+=("$arg") ;;
  esac
done

if [ ! -f "${INPUT_CSS}" ]; then
  echo "Missing input CSS: ${INPUT_CSS}" >&2
  exit 1
fi
if [ ! -f "${CONFIG_FILE}" ]; then
  echo "Missing tailwind config: ${CONFIG_FILE}" >&2
  exit 1
fi

if ! command -v npx >/dev/null 2>&1; then
  echo "npx not found. Install Node.js (>= 18) and try again." >&2
  echo "On Ubuntu:  sudo apt install nodejs npm" >&2
  exit 1
fi

CMD=(npx --yes tailwindcss@3
     -c "${CONFIG_FILE}"
     -i "${INPUT_CSS}"
     -o "${OUTPUT_CSS}"
     --minify)
if [ "${WATCH}" -eq 1 ]; then
  CMD+=(--watch)
fi
CMD+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

echo "==> Building Tailwind CSS"
echo "    config : ${CONFIG_FILE}"
echo "    input  : ${INPUT_CSS}"
echo "    output : ${OUTPUT_CSS}"
(cd "${FRONTEND_DIR}" && "${CMD[@]}")

if [ "${WATCH}" -eq 0 ]; then
  echo "==> Done. Generated $(wc -c < "${OUTPUT_CSS}" | tr -d ' ') bytes."
fi
