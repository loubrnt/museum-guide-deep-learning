# Reproducible environment for the museum guide: Python 3.11, CPU-only PyTorch, known-good versions.
#
#   docker build -t museum-guide .
#   docker run --rm museum-guide                      # runs the test suite
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" \
#       -v "$HOME/.cache/huggingface:/home/app/.cache/huggingface" museum-guide python scripts/baseline_classical.py
#
# The data (data/) and the model downloads are NOT baked into the image: mount them as above.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

# CPU-only PyTorch first: the default Linux wheel pulls about 2 GB of CUDA libraries the project never uses.
# The constraints file keeps it in the tested minor version.
COPY constraints/ constraints/
RUN pip install -c constraints/py311.txt torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Dependencies before the sources, so that editing code does not reinstall them.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install -c constraints/py311.txt -e ".[dev]"

COPY . .
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

CMD ["python", "-m", "pytest", "-q"]
