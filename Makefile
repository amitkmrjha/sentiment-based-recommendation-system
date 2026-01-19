install:
	uv pip install -r requirements.txt
	python -m ipykernel install --user --name capstone-env --display-name "Capstone (env)"