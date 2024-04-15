import os
import requests
import re
from bs4 import BeautifulSoup

def get_urls(url):
    """ Extract urls from the url page passed"""

    response = requests.get(url, timeout=50)
    soup = BeautifulSoup(response.content, 'html.parser')
    url_links = soup.find_all('a', href=lambda href: href and href.startswith('/newsevents/testimony/'))

    urls = [link['href'] for link in url_links]
    return urls


def extract_text_from_url(url):
    # Parse the HTML content using BeautifulSoup
    try:
        response = requests.get(url, timeout=50)
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    content = response.content
    soup = BeautifulSoup(content, 'html.parser')
    # Find the main content element
    #main_content = soup.find('div', id='article') # for the FOMC minutes <div id="article" class="col-xs-12 col-sm-8 col-md-9">
    main_content = soup.findAll('p')
    article_content = ""
    for p in main_content:

        article_content += p.get_text(separator='\n', strip=True)

    # Get the text from the main content element
    #text = main_content.get_text(separator='\n', strip=True)
    return article_content
                             


# Function to scrape and save the text files

# Dynamic URL input
fomc_url = "https://www.federalreserve.gov/newsevents/testimony.htm"

# Define the path to save the scraped data
save_directory = "testimonies"

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Scrape the FOMC data and save it locally
list_urls = get_urls(fomc_url)
for year_url in list_urls:
    print(f"Year: {year_url}")

    year = year_url.split('/')[3].split('-')[0]

    year = year[0:4]

    if year not in ["2016", "2015", "2014", "2013", "2012", "2011"]:
        print(f"Year {year} not supported")
        continue


    year_articles_url = "https://www.federalreserve.gov" + year_url

    article_urls = get_urls(year_articles_url)

    for article in article_urls:
        url = "https://www.federalreserve.gov" + article

        file_name = url.split('/')[-1]

        file_date = file_name.split('.')[0]

        article_name = file_date + ".txt"

        if article_name in os.listdir(save_directory):
            print(f"Already saved: {file_name}")
            break


        text = extract_text_from_url(url)
        with open(f'{save_directory}/{file_date}.txt', 'w', encoding='utf8') as file:
            file.write(text)
        print(f"Saved: {file_name}")    


    