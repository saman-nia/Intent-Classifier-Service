# Intent Classifier Service

A FastAPI-based HTTP service for intent classification using a simple neural network trained on ATIS.

## Project Structure

```
intent-classifier-service/
├── .gitignore
├── hatch.toml
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env
├── README.md
├── data/
│   └── atis/
│       ├── train.tsv
│       └── test.tsv
├── models/
│   ├── model.pth
│   ├── vocab.json
│   └── labels.json
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── schemas.py
│   │       └── endpoints.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── base_classifier.py
│   │   └── intent_classifier.py
│   └── utils/
│       ├── __init__.py
│       └── logger.py
├── trains/
│   ├── train_baseline.py
│   └── train_bert.py
└── tests/
    ├── test_ready.py
    └── test_intent.py
```

## Setup

1. **Pin & activate Python 3.10.12 (via pyenv)**  

   ```bash
   pyenv install -s 3.10.12
   pyenv local 3.10.12
   ```

2. **Point Hatch at that interpreter**  

   ```bash
   export HATCH_PYTHON="$(pyenv which python)"
   ```

3. **(Re)build your virtual env with ALL deps from pyproject.toml**  

   ```bash
   hatch env remove default || true
   hatch env create
   ```

4. **Regenerate the lockfile for reproducibility**  

   ```bash
   hatch run poetry lock
   ```

5. **Train BOTH models in one go**  

   ```bash
   chmod +x scripts/train_all.sh
   ./scripts/train_all.sh
   ```

## API: Two ways, Docker and Hatching

## Docker

1. **Choose which to serve in the docker-compose.yml**

```bash
environment:
      MODEL_IMPL: bert  # or fasttext
      MODEL_PATH: /app/models/bert  # or /app/models/fasttext
```

12. **Build the image**

```bash
docker build -t intent-classifier-service .
```

3. **Run the container using Compose**

```bash
docker-compose up --build -d
```

## From Hatching

1. **Choose which to serve**

### Baseline

```bash
export MODEL_IMPL=fasttext
export MODEL_PATH=models/fasttext
```

### BERT

```bash
export MODEL_IMPL=bert
export MODEL_PATH=models/bert
```

2. **Start the service**

```bash
hatch run start
```

- **GET** `/ready`  
  - 200 `"OK"` if model loaded  
  - 423 `"Not ready"` otherwise

- **POST** `/intent`  

  ```bash
  curl http://localhost:8000/ready
  curl -X POST http://localhost:8000/intent \
       -H 'Content-Type: application/json' \
       -d '{"text":"Find me a flight from Memphis to Tacoma"}'
  ```

  ```json
   {
   "intents": [
      {
         "label": "flight",
         "confidence": 0.9986798167228699
      },
      {
         "label": "airfare",
         "confidence": 0.00015791608893778175
      },
      {
         "label": "airline",
         "confidence": 0.00015062351303640753
      }
   ]
   }
  ```
