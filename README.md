# Federal Reserve Speeches Scraper and Analysis

This repository contains a Python script for downloading and processing speeches from the Federal Reserve website. The main script, `fomc_database.py`, sends GET requests to specific URLs and parses the responses using BeautifulSoup.

# Tokenizer and Sentiment Analysis

The preliminary_analysis.py has two main purposes: tokenizing and evaluating the text database based on the available dictionaries

## Getting Started

To run the script, you'll need Python and the following libraries:

- requests
- BeautifulSoup
- re (regular expressions)

## Usage

To run the script, use the following command:

```bash
python fomc_database.py
```



This will start the process of downloading and parsing the speeches. If a speech fails to download, the script will print a message and continue with the next speech.



License
This project is licensed under the MIT License.
