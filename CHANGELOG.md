# Changelog

All notable changes to **credit-risk-prediction-ml-pipeline** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-22

### Added
- Modular threshold optimization & financial impact reporting module (`src/credit_risk/reporting.py`), reducing monolith ratio below 10%.
- Pinned dependencies in `requirements.txt`, `requirements-lock.txt`, `uv.lock`, and Ruff configuration in `pyproject.toml`.
- Expanded CI pipeline (`.github/workflows/ci.yml`) with Ruff lint, Pytest matrix, Docker build, Semgrep SAST, and Trivy container scanning.
- Security and operations tooling: `.github/dependabot.yml`, `SECURITY.md`, `Makefile`, `docs/runbook.md`, `LICENSE`, `CONTRIBUTING.md`, and `.pre-commit-config.yaml`.

### Changed
- Added structured leveled logging across `src/credit_risk/api.py`, `src/credit_risk/service.py`, `src/credit_risk/reporting.py`, and `credit_risk_prediction.py`.
