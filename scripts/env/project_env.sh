#!/usr/bin/env bash

# Source this file before running project scripts:
#   source scripts/project_env.sh
#
# It keeps plotting/font caches inside this project so Matplotlib does not try
# to write into user-level cache directories that may be unavailable.

set -eu

if [ -n "${BASH_SOURCE:-}" ]; then
  SCRIPT_PATH="${BASH_SOURCE}"
elif [ -n "${ZSH_VERSION:-}" ]; then
  SCRIPT_PATH="${(%):-%N}"
else
  SCRIPT_PATH="$0"
fi

SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_ROOT}/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/.cache}"

mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}/fontconfig"

printf 'Project runtime cache configured:\n'
printf '  MPLCONFIGDIR=%s\n' "${MPLCONFIGDIR}"
printf '  XDG_CACHE_HOME=%s\n' "${XDG_CACHE_HOME}"
