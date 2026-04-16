# Variables
PYTHON = python3
PIP = pip3
IMAGE_NAME = risk-management-bny
LEGACY_SCRIPTS = archive/legacy_etl/scripts
LEGACY_NEWS = archive/legacy_etl/news_feeds

# -- Local Development --

.PHONY: install
install:
	$(PIP) install -r requirements.txt
	$(PIP) install --no-deps pygooglenews

.PHONY: run-key-pipeline
run-key-pipeline:
	$(PYTHON) key_pipeline/run_key_pipeline.py

.PHONY: run-gnews
run-gnews:
	$(PYTHON) $(LEGACY_SCRIPTS)/gnews_etl.py

.PHONY: run-yahoo
run-yahoo:
	$(PYTHON) $(LEGACY_SCRIPTS)/yfinance_etl.py

.PHONY: run-merge
run-merge:
	$(PYTHON) $(LEGACY_SCRIPTS)/combine_news.py

# Legacy Yahoo + GNews + combiner (used by Dockerfile CMD)
.PHONY: run-etl
run-etl: run-gnews run-yahoo run-merge
	@echo "Legacy ETL finished. Outputs under data/ and $(LEGACY_NEWS)."

.PHONY: clean
clean:
	rm -rf data/state/*.csv
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# -- Docker / Production --
.PHONY: docker-build
docker-build:
	docker build -t $(IMAGE_NAME) .

# Runs the container and mounts your local 'data' folder
.PHONY: docker-run
docker-run:
	docker run --rm -v $(PWD)/data:/app/data $(IMAGE_NAME)
