format:
	python -m black .
	python -m isort --profile black .

lint:
	python -m flake8 .

test:
	python -m pytest tests
