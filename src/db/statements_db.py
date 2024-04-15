import os
import requests
import re
from bs4 import BeautifulSoup

def get_urls(url, regex):
    """ Extract urls from the url page passed"""

    response = requests.get(url, regex, timeout=50)
    soup = BeautifulSoup(response.content, 'html.parser')
    url_links = soup.find_all('a',href=lambda href: href and re.match(regex, href, re.IGNORECASE))
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
    #For older than 2005
    ####main_content = soup.find('div', id='article') # for the FOMC minutes <div id="article" class="col-xs-12 col-sm-8 col-md-9">
    
    
    main_content = soup.findAll('p')
    article_content = ""
    for p in main_content:
        article_content += p.get_text(separator='\n', strip=True)
    #main_content = soup.find_all('p')
    #article_content = ""
    #for p in main_content:	
    #    article_content += p.get_text(separator='\n', strip=True)

    # Get the text from the main content element
    #text = main_content.get_text(separator='\n', strip=True)
    return article_content
                             



# Dynamic URL input
fomc_url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

# Define the path to save the scraped data
save_directory = "statements"


def scrape_statements(save_directory, fomc_url):

    # Create the directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    statement_urls = get_urls(fomc_url)
    for url in statement_urls:
        if re.match(r"/newsevents/pressreleases/monetary\d{8}[a-z].htm", url, re.IGNORECASE):
            url = f"https://www.federalreserve.gov{url}"
            text = extract_text_from_url(url)
            with open(f'{save_directory}/{url.split("/")[-1]}.txt', 'w', encoding='utf8') as file:
                file.write(text)
                print(f"Saved: {url.split('/')[-1]}")
        elif re.match(r"/newsevents/pressreleases/2\d{3}all.htm", url, re.IGNORECASE):
            url = f"https://www.federalreserve.gov{url}"

            text = extract_text_from_url(url)
            with open(f'{save_directory}/{url.split("/")[-1]}.txt', 'w', encoding='utf8') as file:
                file.write(text)
                print(f"Saved: {url.split('/')[-1]}")
        else:
            print(f"Skipping: {url}")

    

def get_historical_statements(save_directory):
    for value in ["02", "01", "00"]:
        main_url = f"https://www.federalreserve.gov/monetarypolicy/fomchistorical20{value}.htm"

        if int(value) in range(6, 24):
            regex_year = r"/newsevents/press/monetary/\d{8}[a-z].htm"

        elif int(value) in range(3, 6):
            regex_year = r"/boarddocs/press/monetary/\d{4}/\d{8}/default.htm"

        else:
            if value == "00":
                regex_year = r"/boarddocs/press/general/2000/"
            if value == "01":
                regex_year = r"/boarddocs/press/general/2001/"

            if value == "02":
                regex_year = r"/boarddocs/press/monetary/2002/"
            
        article_links = get_urls(main_url, regex_year)



        for article in article_links:

            article = "https://www.federalreserve.gov" + article

            year = int(value) + 2000

            if year >= 2006:
                date = article.split('/')[-1].split('.')[0]

                savename = f'{date}'
            else:
                date = article.split('/')[-2]

                savename = f'{date}'
            already_saved = os.listdir(save_directory)

            if savename in already_saved:
                    break
            text = extract_text_from_url(article)

            with open(f'{save_directory}/{savename}.htm.txt', 'w', encoding='utf8') as file:
                file.write(text)
                print(f"Saved: {savename}")
                    
# Scrape the FOMC data and save it locally
get_historical_statements(save_directory)
#https://www.federalreserve.gov/monetarypolicy/fomchistorical20{value}.htm

