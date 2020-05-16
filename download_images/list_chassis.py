import os

import requests
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options

path = '/usr/local/bin/chromedriver'
WINDOW_SIZE = "1920,1080"
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
chrome_options.add_argument('--no-sandbox')
# driver = webdriver.Chrome(executable_path=path)

# This function searchs Wikipedia for the information of the teams of each year.
# Then creates a file containing a dictionary with the results
def list_chassis_per_season():
    wikipedia = 'https://en.wikipedia.org'
    url = 'https://en.wikipedia.org/wiki/Category:Formula_One_cars_by_season'
    req = requests.get(url)
    soup = BeautifulSoup(req.content, 'html.parser')
    all_links = soup.find_all('a')
    match = ['Formula', 'One', 'season', 'cars']

    dict_chassis_season={}
    for link in all_links:
        if str(link.get('href')).split('_')[-4:] == match:
            link_season = str(wikipedia+link.get('href')).replace(' ','')
            season = str(link.get('href')).replace(':','_').split('_')[-5:-4][0]
            if int(season)<2021:
                req = requests.get(link_season)
                soup = BeautifulSoup(req.content, 'html.parser')

                teams_season = soup.findAll('div', attrs={'class': 'mw-category-group'})
                teams_season_list = []
                for teams_div in teams_season:
                    teams_links = teams_div.findAll('a')
                    for link in teams_links:
                        teams_season_list.append(link.get('title'))
                dict_chassis_season[season]=teams_season_list
    with open('chassis_season.txt', 'w') as file:
        file.write(str(dict_chassis_season))


# Checks wether is necessary to run 'list_chassis_per_year' function
if not os.path.isfile('chassis_season.txt'):
    list_chassis_per_season()

