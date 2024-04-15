import os
import requests
import re
from bs4 import BeautifulSoup

# Function to scrape and save the text files
def scrape_fomc_data(url, save_path):
    # Send GET request to the URL
    try:
        response = requests.get(url, timeout=5)
    except Exception as e:
        print(f"An error occurred: {e}")
        return  # Exit the function
    
    # Parse the HTML content using BeautifulSoup
    soup_fed = BeautifulSoup(response.content, 'html.parser')
    
    # find all the links that match the desired format

    links = soup_fed.find_all('a', href=lambda href: href and

    re.match(r"/newsevents/speech/", href, re.IGNORECASE))
    
    # Loop through each link
    for link in links:

        url_speach = "https://www.federalreserve.gov" + link['href']

        response_yearly = requests.get(url_speach, timeout=5)

        if response_yearly.status_code != 200:
            print(f"Failed to download: {url_speach}")
            continue

        soup_yearly = BeautifulSoup(response_yearly.content, 'html.parser')

        speaches = soup_yearly.find_all('a', href=lambda href: href and re.match(r"/newsevents/speech/", href, re.IGNORECASE))    

    #links is a list of all pages with speeches for that year
        for speech in speaches:
                href = speech.get('href')
                if href:
                    href = "https://www.federalreserve.gov/" + href
                    # Send GET request to the file URL
                    data_response = requests.get(href, timeout=5)

                    if data_response.status_code != 200:
                        print(f"Failed to download: {href}")
                        continue
                    
                    # Extract the filename from the URL
                    filename = href.split('/')[-1]

                    link_path = link['href'].split('/')[-1]
                    year = link_path.split('-')[0]

                    soup_article = BeautifulSoup(data_response.content, 'html.parser')

                    main_content = soup_article.find('div', id='article') # for the FOMC minutes <div id="article" class="col-xs-12 col-sm-8 col-md-9">
        
        # Get the text from the main content element
    
                    if main_content is not None:
                        text = main_content.get_text(separator='\n', strip=True)
                    else:
                        print("Div not found")

                    path = os.path.join(save_path, filename)

                    
                    #    Remove links
                    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

                    # Save the content as a text file
                    with open(f'{path}.txt', 'w', encoding="utf-8") as file:
                        file.write(text)
                        print(f"Saved: {filename}")

# Dynamic URL input
fomc_url = "https://www.federalreserve.gov/newsevents/speeches.htm"

# Define the path to save the scraped data
save_directory = "speeches"

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Scrape the FOMC data and save it locally
scrape_fomc_data(fomc_url, save_directory)