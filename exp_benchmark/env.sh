# Activate conda env with hashannlib installed.
# Override env name with EMA_CONDA_ENV.
EMA_CONDA_ENV="${EMA_CONDA_ENV:-py310}"

# Try common conda locations
for cand in \
    "/home/mocheng/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -f "$cand" ]; then
        # shellcheck disable=SC1090
        source "$cand"
        conda activate "$EMA_CONDA_ENV"
        return 0 2>/dev/null || true
        break
    fi
done

# pybind11 false-positive heap-corruption guard
export MALLOC_CHECK_=3
