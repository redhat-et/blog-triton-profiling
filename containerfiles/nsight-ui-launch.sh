#!/bin/bash
# Launch an Nsight UI outside the workspace venv.
#
# Nsight Systems/Compute embed CPython and reinitialize it while closing
# the UI. That interpreter scans site-packages .pth files, and its reduced
# codec registry cannot decode setuptools' distutils-precedence.pth as
# utf-8-sig. The image removes the base-image copy; this launcher keeps the
# embedded interpreter away from the workspace venv's copy (and from the
# activation hooks in the image environment).
#
# Installed as /usr/local/bin/nsight-ui-launch and called by the nsys-ui /
# ncu-ui wrappers, so the make targets, `docker run <image> nsys-ui`, and
# an interactive shell all take the same path.

set -o errexit
set -o nounset
set -o pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: nsight-ui-launch <ui-binary> [args...]" >&2
    exit 2
fi

ui_cmd=$1
shift
ui_pwd=$PWD

# The UIs open relative .ncu-rep/.nsys-rep paths against their working
# directory, and this script changes to / below. Resolve report arguments
# against the caller's directory first; other non-option arguments pass
# through unchanged.
ui_args=()
for arg in "$@"; do
    case "$arg" in
        [!-]*.ncu-rep | [!-]*.nsys-rep)
            case "$arg" in
                /*) ;;
                *) arg="${ui_pwd}/${arg}" ;;
            esac
            ;;
    esac
    ui_args+=("$arg")
done

# Drop the venv's bin directory from PATH; the activation hooks are unset
# in the exec below.
ui_path=""
old_ifs=$IFS
IFS=:
for path_dir in $PATH; do
    [ "$path_dir" = "${WORKSPACE:-/workspace}/bin" ] && continue
    ui_path="${ui_path:+${ui_path}:}${path_dir}"
done
IFS=$old_ifs

cd /
exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
    -u BASH_ENV -u ENV -u PROMPT_COMMAND \
    PATH="$ui_path" "$ui_cmd" "${ui_args[@]}"
