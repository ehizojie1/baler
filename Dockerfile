# Based on https://github.com/michaeloliverx/python-poetry-docker-example/blob/master/docker/Dockerfile

## -----------------------------------------------------------------------------
## Base image with VENV

FROM python:3.8-slim as python-base

# Configure environment

ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/baler-root/baler" \
    VENV_PATH="/baler-root/baler/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# System deps:
RUN pip install "poetry"

# Copy only requirements to cache them in docker layer
WORKDIR $PYSETUP_PATH
COPY ./poetry.lock ./pyproject.toml ./

# Project initialization:
RUN poetry install --no-interaction --no-ansi

# Creating folders, and files for the project:
COPY ./baler/ __init__.py README.md ./tests/ ./

# Creating python wheel
RUN poetry build

## -----------------------------------------------------------------------------
## Baler layer

FROM python:3.8-slim

# Copy virtual environment
WORKDIR /baler-root/baler
COPY --from=python-base /baler-root/baler/modules/ ./modules
COPY --from=python-base /baler-root/baler/*.py /baler-root/baler/README.md ./
COPY --from=python-base /baler-root/baler/dist/*.whl ./

# Install wheel
RUN pip install *.whl

# Configure run time
WORKDIR /baler-root/
ENTRYPOINT ["python", "baler"]
