import os
from bs4 import BeautifulSoup


def extract_text_from_html(html):
    # Parse the HTML content using BeautifulSoup

    with open(html, 'r', encoding='utf8') as file:
        content = file.read()
        
    soup = BeautifulSoup(content, 'html.parser')
        
        # Find the main content element
    main_content = soup.find('div', id='article', class_='col-xs-12 col-sm-8 col-md-9') # for the FOMC minutes <div id="article" class="col-xs-12 col-sm-8 col-md-9">
   # main_content = soup.find('div', class_='col-xs-12 col-sm-8 col-md-8') # for the monetary press releases. speeches

        
        # Get the text from the main content element
    
    text = main_content.get_text(separator='\n', strip=True)
        
    return text

# Example usage
htmls = os.listdir('fomc_data')

already_done = os.listdir('fomc_text')

todo = [html for html in htmls if html + '.txt' not in already_done]


for file in todo:
    try:
        text = extract_text_from_html(os.path.join('fomc_data', file))
        with open(f'fomc_text/{file}.txt', 'w', encoding='utf8') as file:
            file.write(text)
    except Exception as e:
        print(f'Error with {file}: {str(e)}')
        pass