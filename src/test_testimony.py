from selenium import webdriver
from bs4 import BeautifulSoup
import time
# Set up Chrome WebDriver
driver = webdriver.Chrome()
url = "https://www.federalreserve.gov/newsevents/testimony.htm"
driver.get(url)

# Extract the HTML content
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")

time.sleep(2)


# Find all the testimony links
testimony_links = soup.find_all("span", class_="itemTitle")


# Extract the testimony titles and URLs
for link in testimony_links:
    testimony_title = link.find("a").text
    testimony_url = link.find("a")["href"]
    print(f"Title: {testimony_title}\nURL: {testimony_url}\n")

# Close the browser
driver.quit()
driver.quit()