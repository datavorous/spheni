set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

BUILD_PYTHON=0
INSTALL_PREFIX=""

usage() {
    echo "Usage: $0 [--python] [--install <prefix>]"
    echo "  --python           Build Python module as well"
    echo "  --install <prefix> Run cmake --install to the given prefix"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            BUILD_PYTHON=1
            shift
            ;;
        --install)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --install"
                exit 1
            fi
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

CMAKE_ARGS=(-S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release)
if [[ "${BUILD_PYTHON}" -eq 1 ]]; then
    CMAKE_ARGS+=(-DSPHENI_BUILD_PYTHON=ON)
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}"

if [[ -n "${INSTALL_PREFIX}" ]]; then
    cmake --install "${BUILD_DIR}" --prefix "${INSTALL_PREFIX}"
    echo "Installed to: ${INSTALL_PREFIX}"
fi

echo "Built static library at: ${BUILD_DIR}/libspheni.a"
if [[ "${BUILD_PYTHON}" -eq 1 ]]; then
    echo "Built Python module at: ${BUILD_DIR}/_core*.so"
fi
