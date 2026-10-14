#!/bin/sh
# Run as root just long enough to make the bind-mounted data dir writable, then drop to appuser.
set -e
mkdir -p "${RAG_DATA_DIR:-/data}"
if [ "$(id -u)" = "0" ]; then
    chown -R appuser:appuser "${RAG_DATA_DIR:-/data}"
    exec gosu appuser "$@"
fi
exec "$@"
