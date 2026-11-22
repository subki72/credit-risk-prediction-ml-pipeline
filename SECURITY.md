# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in **Credit Risk Prediction ML Pipeline**:

1. **Do NOT open a public GitHub issue** for undisclosed vulnerabilities.
2. Submit a private report via GitHub Security Advisories (`Security -> Advisories -> Report a vulnerability`).
3. Include affected files, reproduction steps, and estimated severity.

## Security Controls

- **Model Integrity Protection**: Serialized artifacts are verified via SHA-256 checksums (`verify_and_load_artifact`) before loading via `joblib`.
- **Strict Schema Validation**: Pydantic v2 enforces numerical bounds (`loan_amnt > 0`, valid credit grades) rejecting malformed requests with HTTP `422`.
- **Automated Scanning**: Dependabot weekly alerts, Semgrep SAST, and Trivy container vulnerability scans run on every push and PR.
- **Container Hardening**: Multi-stage Docker build with non-root runtime user.
