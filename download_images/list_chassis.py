import json
import os
import re
import time
import urllib.request as ulib

import requests
from bs4 import BeautifulSoup
from selenium import webdriver

json_file = 'chassis_season.json'


# This function searchs Wikipedia for the information of the teams of each year.
# Then creates a file containing a dictionary with the results
def list_chassis_per_season():
    wikipedia = 'https://en.wikipedia.org'
    url = 'https://en.wikipedia.org/wiki/Category:Formula_One_cars_by_season'
    req = requests.get(url)
    soup = BeautifulSoup(req.content, 'html.parser')
    all_links = soup.find_all('a')
    match = ['Formula', 'One', 'season', 'cars']
    dict_chassis_season = {}

    for link in all_links:
        if str(link.get('href')).split('_')[-4:] == match:
            link_season = str(wikipedia + link.get('href')).replace(' ', '')
            season = str(link.get('href')).replace(':', '_').split('_')[-5:-4][0]
            if int(season) < 2021:
                req = requests.get(link_season)
                soup = BeautifulSoup(req.content, 'html.parser')
                teams_season = soup.findAll('div',attrs={'class': 'mw-category-group'})
                teams_season_list = []

                for teams_div in teams_season:
                    teams_links = teams_div.findAll('a')
                    for link in teams_links:
                        teams_season_list.append(link.get('title'))
                dict_chassis_season[season] = teams_season_list

    json_dict = json.dumps(dict_chassis_season)

    with open(json_file, 'w') as file:
        file.write(json_dict)


# This function creates folders to store the photos of each car
def create_folders():
    folder = 'photos'
    with open(json_file, 'r') as file:
        data = file.read()
    loaded_json = json.loads(data)
    if not os.path.isdir(folder):
        os.mkdir(folder)

    for k, v in loaded_json.items():
        if not os.path.isdir(os.path.join(folder, k)):
            os.mkdir(os.path.join(folder, k))

        for team in v:
            team = re.sub(r"[^a-zA-Z0-9]+", ' ', team)
            if not os.path.isdir(os.path.join(folder, k, k+' - '+team)):
                os.mkdir(os.path.join(folder, k, k+' - '+team))


def download_photos(amount, start_year, stop_year):
    with open(json_file, 'r') as file:
        data = file.read()
    loaded_json = json.loads(data)

    path = '/home/alexandresalem/Projects/f1car_image_classifier/download_images/chromedriver'
    # WINDOW_SIZE = "1920,1080"
    # chrome_options = Options()
    # chrome_options.add_argument("--headless")
    # chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
    # chrome_options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(executable_path=path)

    for k in loaded_json.keys():
        if int(k) in range(start_year, stop_year+1):
            chassis_season = os.listdir(os.path.join('photos', k))

            for chassis in chassis_season:
                url = f'https://www.google.com/search?tbm=isch&q={chassis}'
                driver.get(url)
                time.sleep(0.5)
                download_list = []
                download_list.clear()

                for el in range(amount):
                    try:
                        driver.find_element_by_xpath(f'/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[1]/div[1]/div[{el+1}]/a[1]/div[1]/img').click()
                        time.sleep(0.5)
                        img = driver.find_element_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img').get_property('src')
                        if img[-3:].lower() in ['jpg', 'png']:
                            download_list.append(img)
                    except:
                        pass

                if len(download_list) > 0:
                    current_folder = os.path.join('photos', k, chassis)
                    current_photos = os.listdir(current_folder)
                    last_id = 0
                    for id, photo in enumerate(current_photos):
                        old=photo
                        new=str(id+100)+photo[-4:]
                        try:
                            os.rename(os.path.join(current_folder, old),
                                      os.path.join(current_folder, new))
                        except:
                            pass

                        last_id = id

                    for item, link in enumerate(download_list):
                        path = os.path.join('photos', k, chassis,str(last_id+item+101)+link[-4:])

                        try:
                            ulib.urlretrieve(link, path)
                            print(f'Arquivo {path} salvo!')
                        except:
                            print('Failed ULIB')


# Checks wether is necessary to run 'list_chassis_per_year' function
# if not os.path.isfile(json_file):
#     list_chassis_per_season()
# create_folders()
download_photos(100, 2020, 2020)
