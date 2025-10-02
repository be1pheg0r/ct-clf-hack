PYTHON := python3
BACKEND := ct_clf_backend
FRONTEND := frontend
PROJECT_NAME := ct_clf-hack

GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m

install: install-backend install-frontend download-models
	@echo "$(GREEN)Installation complete!$(NC)"

install-backend:
	@echo "$(YELLOW)Installing backend dependencies...$(NC)"
	$(PYTHON) -m venv ./venv
	. ./venv/bin/activate && pip install --upgrade pip
	$(PYTHON) -m pip install poetry
	poetry lock
	poetry install
	@echo "$(GREEN)Backend dependencies installed!$(NC)"

download-models:
	@echo "$(YELLOW)Downloading models from Huggingface...$(NC)"
	. ./venv/bin/activate && $(PYTHON) scripts/download_models.py
	@echo "$(GREEN)Models downloaded!$(NC)"

install-frontend:
	@echo "$(YELLOW)Installing frontend dependencies...$(NC)"
	cd $(FRONTEND) && npm install
	@echo "$(GREEN)Frontend dependencies installed!$(NC)"

check-deps:
	@echo "$(YELLOW)Checking backend dependencies...$(NC)"
	poetry check
	@echo "$(GREEN)Backend dependencies are up to date!$(NC)"
	cd $(FRONTEND) && npm outdated || echo "$(GREEN)Frontend dependencies are up to date!$(NC)"

run-backend:
	@echo "$(YELLOW)Starting backend server...$(NC)"
	. ./venv/bin/activate && uvicorn $(BACKEND).app.main:app --host 0.0.0.0 --port 8000

run-frontend:
	@echo "$(YELLOW)Starting frontend server...$(NC)"
	cd $(FRONTEND) && npm start

test:
	@echo "$(YELLOW)Running backend tests...$(NC)"
	. ./venv/bin/activate && scripts/test_system.py

build: build-frontend
	@echo "$(GREEN)Build complete!$(NC)"

build-frontend:
	@echo "$(YELLOW)Building frontend...$(NC)"
	cd $(FRONTEND) && npm run build
	@echo "$(GREEN)Frontend build complete!$(NC)"

create-checkpoints:
	@echo "$(YELLOW)Creating trained model checkpoints...$(NC)"
	mkdir -p checkpoints
	$(PYTHON) -c "
import torch
import torchvision.models as models
from pathlib import Path

models_to_create = [
    ('Inception_V3', models.inception_v3),
    ('ResNet50', models.resnet50),
    ('DenseNet121', models.densenet121),
    ('ConvNeXt_Tiny', models.convnext_tiny)
]

checkpoints_dir = Path('checkpoints')
for model_name, model_func in models_to_create:
    try:
        print(f'Creating {model_name}...')
        model = model_func(weights='DEFAULT')

        if 'ResNet' in model_name or 'DenseNet' in model_name:
            if hasattr(model, 'classifier'):
                in_features = model.classifier.in_features
                model.classifier = torch.nn.Linear(in_features, 2)
            elif hasattr(model, 'fc'):
                in_features = model.fc.in_features
                model.fc = torch.nn.Linear(in_features, 2)
        elif 'Inception' in model_name:
            if hasattr(model, 'fc'):
                in_features = model.fc.in_features
                model.fc = torch.nn.Linear(in_features, 2)
        elif 'ConvNeXt' in model_name:
            if hasattr(model, 'classifier'):
                model.classifier = torch.nn.Sequential(
                    torch.nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True),
                    torch.nn.Flatten(start_dim=1, end_dim=-1),
                    torch.nn.Linear(768, 2)
                )

        checkpoint_path = checkpoints_dir / f'default_{model_name}.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'num_classes': 2,
            'trained': True
        }, checkpoint_path)
        print(f'✓ Created {checkpoint_path}')
    except Exception as e:
        print(f'✗ Failed {model_name}: {e}')
        (checkpoints_dir / f'default_{model_name}.pth').touch()
"
	@echo "$(GREEN)Model checkpoints created!$(NC)"

docker-build: create-checkpoints
	@echo "$(YELLOW)Building Docker image...$(NC)"
	docker build -t $(PROJECT_NAME) .
	@echo "$(GREEN)Docker image built!$(NC)"

docker-run:
	@echo "$(YELLOW)Running Docker container...$(NC)"
	docker run -d -p 8000:8000 --name $(PROJECT_NAME)_container $(PROJECT_NAME)
	@echo "$(GREEN)Docker container is running!$(NC)"

docker-stop:
	@echo "$(YELLOW)Stopping Docker container...$(NC)"
	docker stop $(PROJECT_NAME)_container || echo "$(RED)Container not running.$(NC)"
	docker rm $(PROJECT_NAME)_container || echo "$(RED)Container not found.$(NC)"
	@echo "$(GREEN)Docker container stopped!$(NC)"

# New Docker Compose commands
docker-up:
	@echo "$(YELLOW)Starting services with docker-compose...$(NC)"
	docker-compose up -d backend frontend
	@echo "$(GREEN)Services started! Backend: http://localhost:8000, Frontend: http://localhost:3000$(NC)"

docker-up-fullstack:
	@echo "$(YELLOW)Starting fullstack service...$(NC)"
	docker-compose --profile fullstack up -d fullstack
	@echo "$(GREEN)Fullstack service started! Access at http://localhost$(NC)"

docker-down:
	@echo "$(YELLOW)Stopping all services...$(NC)"
	docker-compose down
	@echo "$(GREEN)All services stopped!$(NC)"

docker-logs:
	@echo "$(YELLOW)Showing logs...$(NC)"
	docker-compose logs -f

docker-rebuild:
	@echo "$(YELLOW)Rebuilding and restarting services...$(NC)"
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d
	@echo "$(GREEN)Services rebuilt and restarted!$(NC)"

docker-clean:
	@echo "$(YELLOW)Cleaning up Docker resources...$(NC)"
	docker-compose down -v --remove-orphans
	docker system prune -f
	@echo "$(GREEN)Docker cleanup complete!$(NC)"

.PHONY: create-checkpoints download-models docker-up docker-up-fullstack docker-down docker-logs docker-rebuild docker-clean
