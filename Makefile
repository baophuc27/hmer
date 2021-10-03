init:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install

format:
	black . --line-length=120
	isort . --multi-line VERTICAL_HANGING_INDENT --trailing-comma --line-width=120
	flake8

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf lightning_logs/

train:
	python run.py --RUN='train'
