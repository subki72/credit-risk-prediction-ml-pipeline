.PHONY: install test lint format run train docker-build clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	ruff check src tests

format:
	ruff format src tests

run:
	uvicorn src.credit_risk.api:app --host 0.0.0.0 --port 8000 --reload

train:
	python credit_risk_prediction.py

docker-build:
	docker build -t credit-risk-api:latest .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
