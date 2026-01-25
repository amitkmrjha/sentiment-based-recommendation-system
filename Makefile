# Variables
APP_NAME = recommendation-app
DOCKER_USERNAME = amitjha12
TAG = latest
PORT = 8000

install:
	uv pip install -r requirements.txt
	python -m ipykernel install --user --name capstone-env --display-name "Capstone (env)"

# Build Docker image
build:
	docker build -t $(DOCKER_USERNAME)/$(APP_NAME):$(TAG) .

# Run Docker container locally
run:
	docker run -p $(PORT):$(PORT) --name $(APP_NAME)_container $(DOCKER_USERNAME)/$(APP_NAME):$(TAG)

# Stop and remove the running container
stop:
	docker stop $(APP_NAME)_container || true
	docker rm $(APP_NAME)_container || true

# Push image to Docker Hub
push:
	docker push $(DOCKER_USERNAME)/$(APP_NAME):$(TAG)

# Clean all images with this app name
clean:
	docker rmi $(DOCKER_USERNAME)/$(APP_NAME):$(TAG) || true

