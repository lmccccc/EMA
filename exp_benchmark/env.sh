# Keep the caller's Python environment unless a conda environment is requested.
if [ -n "${EMA_CONDA_ENV:-}" ]; then
    if ! declare -F conda >/dev/null; then
        for cand in \
            "$HOME/anaconda3/etc/profile.d/conda.sh" \
            "$HOME/miniconda3/etc/profile.d/conda.sh" \
            "/opt/conda/etc/profile.d/conda.sh"; do
            if [ -f "$cand" ]; then
                # shellcheck disable=SC1090
                source "$cand"
                break
            fi
        done
    fi
    if ! declare -F conda >/dev/null; then
        echo "[env] EMA_CONDA_ENV requires an initialized conda shell" >&2
        return 1
    fi
    if ! conda activate "$EMA_CONDA_ENV"; then
        echo "[env] failed to activate conda environment: $EMA_CONDA_ENV" >&2
        return 1
    fi
fi

# pybind11 false-positive heap-corruption guard
export MALLOC_CHECK_=3
