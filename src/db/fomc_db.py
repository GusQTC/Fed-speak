import os
import requests
import re
from bs4 import BeautifulSoup

def get_urls(url):
    """ Extract urls from the url page passed"""

    response = requests.get(url, timeout=50)
    soup = BeautifulSoup(response.content, 'html.parser')
    url_links = soup.find_all('a',href=lambda href: href and re.match(r"/newsevents/pressreleases/", href, re.IGNORECASE))
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
    main_content = soup.find('div', id='article') # for the FOMC minutes <div id="article" class="col-xs-12 col-sm-8 col-md-9">
    article_content = main_content.get_text(separator='\n', strip=True)
    #main_content = soup.find_all('p')
    #article_content = ""
    #for p in main_content:	
    #    article_content += p.get_text(separator='\n', strip=True)

    # Get the text from the main content element
    #text = main_content.get_text(separator='\n', strip=True)
    return article_content
                             


# Function to scrape and save the text files
def scrape_fomc_data(url, save_path):
    """wORKS AFTER 2018"""
    # Send GET request to the URL
    try:
        response = requests.get(url, timeout=50)
    except Exception as e:
        print(f"An error occurred: {e}")
        return  # Exit the function
    
    if response.status_code != 200:
        print(f"Failed to download: {url}")
        return  # Exit the function
    
    # Parse the HTML content using BeautifulSoup
    soup_fed = BeautifulSoup(response.content, 'html.parser')
    
    # find all the links that match the desired format

    fed_links = soup_fed.find_all('a', href=lambda href: href and
                              
    #https://www.federalreserve.gov/monetarypolicy/fomcminutes20231213.htm

    re.match(r"/monetarypolicy/", href, re.IGNORECASE))
    
    # Loop through each link
    for fed_link in fed_links:

        url_speach = "https://www.federalreserve.gov" + fed_link['href']

        response_yearly = requests.get(url_speach, timeout=50)

        if response_yearly.status_code != 200:
            print(f"Failed to download: {url_speach}")
            continue

        soup_yearly = BeautifulSoup(response_yearly.content, 'html.parser')

        publications = soup_yearly.find_all('a', href=lambda href: href and re.match(r"/monetarypolicy/fomcminutes20[0-1][0-9].htm", href, re.IGNORECASE))

    #links is a list of all pages with speeches for that year
        for article in publications:
                href = article.get('href')
                if href:
                    href = "https://www.federalreserve.gov/" + href
                    # Send GET request to the file URL
                    data_response = requests.get(href, timeout=50)

                    if data_response.status_code != 200:
                        print(f"Failed to download: {href}")
                        continue
                    
                    # Extract the filename from the URL
                    filename = href.split('/')[-1]

                    link_path = article['href'].split('/')[-1]
                    year = link_path.split('-')[0]

                    soup_article = BeautifulSoup(data_response.content, 'html.parser')

                    main_content = soup_article.find('div', id='article') # for the FOMC minutes <div id="article" class="col-xs-12 col-sm-8 col-md-9">
        
        # Get the text from the main content element
    
                    if main_content is not None:
                        article_content = main_content.get_text(separator='\n', strip=True)
                    else:
                        print("Div not found")

                    path = os.path.join(save_path, filename)

                    
                    #    Remove links
                    article_content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', article_content)

                    # Save the content as a text file
                    with open(f'{path}.txt', 'w', encoding="utf-8") as text_file:
                        text_file.write(article_content)
                        print(f"Saved: {filename}")

# Dynamic URL input
fomc_url = "https://www.federalreserve.gov/newsevents/pressreleases.htm"

# Define the path to save the scraped data
save_directory = "monetary"

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

year_urls = get_urls(fomc_url)
for url in year_urls:
    if re.match(r"/newsevents/pressreleases/monetary\d{8}[a-z].htm", url, re.IGNORECASE):
        text = extract_text_from_url(url)
        with open(f'{save_directory}/{url.split("/")[-1]}.txt', 'w', encoding='utf8') as file:
            file.write(text)
            print(f"Saved: {url.split('/')[-1]}")
    elif re.match(r"/newsevents/pressreleases/2\d{3}all.htm", url, re.IGNORECASE):
        text = extract_text_from_url(url)
        with open(f'{save_directory}/{url.split("/")[-1]}.txt', 'w', encoding='utf8') as file:
            file.write(text)
            print(f"Saved: {url.split('/')[-1]}")
    else:
        print(f"Skipping: {url}")
# Scrape the FOMC data and save it locally
#scrape_fomc_data(fomc_url, save_directory)
    


#https://www.federalreserve.gov/monetarypolicy/fomchistorical20{value}.htm

def get_historical_data(save_directory):
    for value in ["18", "17", "16", "15", "14", "13", "12", "11", "10", "09", "08", "07", "06", "05", "04", "03", "02", "01", "00"]:
        main_url = f"https://www.federalreserve.gov/monetarypolicy/fomchistorical20{value}.htm"
        file = f"fomc_historical_{value}.txt"
        year_links = get_urls(main_url)

        for link in year_links:
                
            if re.match(r"/newsevents/pressreleases/monetary\d{8}[a-z].htm", link, re.IGNORECASE):

                link = "https://www.federalreserve.gov" + link
                date = link.split('/')[-1].split('.')[0]
                date = date.split('monetary')[1]
                date = date.split('.')[0]

                savename = f'{date}'
                already_saved = os.listdir(save_directory)

                if savename in already_saved:
                    break
                text = extract_text_from_url(link)

                with open(f'{save_directory}/{savename}.htm.txt', 'w', encoding='utf8') as file:
                    file.write(text)
                    print(f"Saved: {savename}")