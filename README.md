# MEDIQ 🤖

[![Streamlit](https://img.shields.io/badge/Streamlit-orange?logo=streamlit&logoColor=FFAF45)](https://streamlit.io/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-%23F37626.svg?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%2334D058.svg?style=flat&logo=hugging-face&logoColor=white)](https://huggingface.co/)
[![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-%23007ACC.svg?style=flat&logo=visual-studio-code&logoColor=white)](https://code.visualstudio.com/)
[![Transformer](https://img.shields.io/badge/Transformer-%2334D058.svg?style=flat&logo=transformer&logoColor=white)](https://huggingface.co/transformers/)

This project aims to automate the extraction of medical information from images and texts. It leverages various machine learning models and techniques, including Optical Character Recognition (OCR), Named Entity Recognition (NER), and pre-trained Large Language Models (LLMs). The dataset used for training and evaluation is sourced from web scraping medical information.

## Workflow
![Input](1.png)

## Table of Contents

1. [Image to Text using OCR](#image-to-text-using-ocr)
2. [NER Model for Medicine Name Extraction](#ner-model-for-medicine-name-extraction)
3. [Pre-trained LLM Model: Mixtral-8x7B](#pre-trained-llm-model-mixtral-8x7b)
4. [Medicine Dataset from Web Scraping](#medicine-dataset-from-web-scraping)
5. [Fine Tuning The Model](#fine-tuning-the-model)
6. [Setup and Installation](#setup-and-installation)

## Image to Text using OCR

We utilize an OCR model to convert images of medical documents, prescriptions, and labels into text format. This step is crucial for digitizing information that is often only available in printed form.

### Key Features:
- High accuracy in text extraction from images.
- Supports multiple languages and fonts.
- Preprocessing steps to enhance image quality before OCR.

[Link to OCR implementation](https://github.com/mdabucse/Drastic-Innovators-Aventus-2.0/tree/main/Model)

## NER Model for Medicine Name Extraction

After converting images to text, we employ a Named Entity Recognition (NER) model to identify and extract names of medicines from the text. This model is specifically trained to recognize medical terminologies and drug names.

### Key Features:
- Trained on a diverse dataset of medical texts.
- High precision and recall in identifying medicine names.
- Can handle variations in drug naming conventions.
- Related reference link will also be provided for the drug

[Link to NER implementation](https://github.com/mdabucse/Drastic-Innovators-Aventus-2.0/tree/main/Model)

## Pre-trained LLM Model: Mixtral-8x7B

For advanced text processing and understanding, we integrate Mixtral-8x7B, a pre-trained Large Language Model (LLM). This model helps in understanding context, disambiguating entities, and enhancing the overall accuracy of information extraction.

### Key Features:
- 8x7 billion parameters providing extensive language understanding.
- Fine-tuned for medical text processing.
- Supports various NLP tasks beyond NER, such as summarization and question answering.

[Link to Mixtral-8x7B usage](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)

## Medicine Dataset from Web Scraping

The dataset used for training and evaluating our models is obtained through web scraping from reliable medical sources. This dataset includes a wide range of medical documents, drug descriptions, and other relevant texts.

### Key Features:
- Comprehensive dataset covering a wide range of medicines.
- Regularly updated to include the latest information.
- Annotated with entity labels for supervised learning tasks.

### Web Scraping Process:
We utilize web scraping to gather data from various medical websites. The process involves:

1. *Identifying Sources*: Selecting reliable websites that provide comprehensive information about medicines.
2. *Scraping Data*: Using tools like BeautifulSoup and Scrapy to extract data such as drug names, descriptions, usage, side effects, etc.
3. *Data Cleaning*: Processing the scraped data to remove duplicates, correct errors, and standardize formats.
4. *Storing Data*: Saving the cleaned data in a structured format (e.g., CSV, JSON) for use in training and evaluation.

### Tools Used:
- *Requests*: To send HTTP requests to the target website and retrieve the HTML content.
- *BeautifulSoup*: To parse the HTML content and extract relevant information.
- *Scrapy*: For more advanced and large-scale web scraping tasks.

### Example Script for Web Scraping:
Here's a simple example of a web scraping script using Python's BeautifulSoup:

python
import requests
from bs4 import BeautifulSoup
import csv

# URL of the website to scrape
url = 'https://example-medical-website.com/medicines'

# Send an HTTP request to the website
response = requests.get(url)

# Parse the HTML content of the page
soup = BeautifulSoup(response.text, 'html.parser')

# Initialize a list to store the scraped data
medicines = []

# Find and loop through all elements containing medicine information
for item in soup.find_all('div', class_='medicine-item'):
    name = item.find('h2').text
    description = item.find('p', class_='description').text
    medicines.append({'name': name, 'description': description})

# Save the scraped data to a CSV file
with open('medicines.csv', 'w', newline='') as csvfile:
    fieldnames = ['name', 'description']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for medicine in medicines:
        writer.writerow(medicine)
