set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

# Inside the CANN container the build runs as root, where `--user` would install
# into /root/.local and `sudo` is not available. Only use them off-root.
PIP_USER_FLAG="--user"
SUDO="sudo"
if [ "$(id -u)" = "0" ]; then
    PIP_USER_FLAG=""
    SUDO=""
elif ! command -v sudo >/dev/null 2>&1; then
    SUDO=""
fi

# Python deps: only the server-specific packages not already provided by the
# framework stack (see requirements.txt). Already-installed packages are left
# untouched; torch* are intentionally NOT re-pinned here.
pip install -r "${ROOT_DIR}/requirements.txt" ${PIP_USER_FLAG}

# System deps: redis (reward queue) + iproute2 (`ss`, used by auto_configure.sh
# for port probing).
${SUDO} apt-get update
${SUDO} apt-get install -y iproute2 redis
