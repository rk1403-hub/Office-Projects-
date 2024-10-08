
# Project: Anomaly Detection, Software Version Tracking, Image Text Detection, and Ticket Management

## Overview

This project contains several Python scripts and configurations for various functionalities:
1. **Anomaly Detection and Vulnerability Prediction** using regression models.
2. **Software Version Detection** using text classification models.
3. **Text Detection from Images** using Optical Character Recognition (OCR).
4. **Ticket Management** with periodic tasks controlled by cron jobs and processed inside Docker containers.

Each script is structured with modular classes and methods, making them reusable and adaptable for various datasets and applications.

## Prerequisites

All scripts rely on several Python libraries. Below is a list of the libraries required for each script. You can install the dependencies via `pip`:

```bash
pip install pandas scikit-learn pymongo certifi requests beautifulsoup4 nltk joblib Pillow pytesseract tensorflow
```

Additionally, ensure you have **Tesseract-OCR** installed for the OCR functionalities, and **Docker** is configured properly to run the containerized services.

## Folder: `tickets`

This folder contains the following scripts:
- **helpers.py**: This script contains helper functions that assist in various NLP and processing tasks. It’s used in the training and prediction models.
- **nltktraining.py**: This script is responsible for training an NLP model using NLTK and other libraries. It can be scheduled as a cron job for periodic updates.
- **nltkprediction.py**: This script performs predictions using the trained model. Like the training script, it’s also scheduled to run periodically using cron.

## Configuration Files

### `cronfile`
This file defines the cron jobs that periodically execute training and prediction tasks:

```bash
# Run the training job every Tuesday at 12:38 PM
38 12 * * 2 /usr/local/bin/python /usr/app/src/nltktraining.py >> /var/log/cron.log 2>&1

# Run the prediction job every 5 minutes
*/5 * * * * /usr/local/bin/python /usr/app/src/nltkprediction.py >> /var/log/cron.log 2>&1
```

### `Dockerfile`
The Dockerfile sets up a Python environment with all the necessary dependencies, installs cron, and configures it to execute the tasks defined in the `cronfile`.

```Dockerfile
FROM python:latest
WORKDIR /usr/app/src
RUN apt-get update && apt-get install -y cron

COPY helpers.py ./
COPY nltktraining.py ./
COPY nltkprediction.py ./
COPY cronfile /etc/cron.d/cronfile

RUN pip install pandas tensorflow scikit-learn pymongo nltk
RUN chmod 0744 ./nltktraining.py ./nltkprediction.py
RUN chmod a+x /etc/cron.d/cronfile
RUN crontab /etc/cron.d/cronfile
RUN touch /var/log/cron.log
CMD cron && tail -f /var/log/cron.log
```

## How to Use

1. **Install dependencies**: Ensure all required libraries are installed.
2. **Docker Setup**: Build and run the Docker container using the provided `Dockerfile`.
3. **Cron Jobs**: The cron jobs are defined in the `cronfile`, which will automatically schedule and run training and prediction tasks.

Example to build the Docker image:
```bash
docker build -t ticket-processor .
docker run -d ticket-processor
```
