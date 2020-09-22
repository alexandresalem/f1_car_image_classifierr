import json
import os
import re
import time
import urllib.request as ulib

import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from pexpect.ANSI import term
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from constants import WIKIPEDIA, WIKIPEDIA_F1_URL, F1_CHASSIS, TRAIN_FOLDER, PHOTO_AMOUNT, API_KEY, CX


def load_json(file) -> dict:
    with open(file, 'r') as file:
        data = file.read()
    return json.loads(data)


def save_json(dictionary: dict, filename=F1_CHASSIS):
    json_dict = json.dumps(dictionary)
    with open(os.path.realpath(filename), 'w') as file:
        file.write(json_dict)


def list_chassis_per_season(start_year=1950, end_year=2020, file_changed=False):
    """
    Search Wikipedia for a list of every F1 Constructor ever and updates JSON file
    :return: Dict containing every F1 chassis of every year
    """

    def _get_chassis_names(years):
        req = requests.get(WIKIPEDIA_F1_URL)
        soup = BeautifulSoup(req.content, 'html.parser')
        links = soup.find_all('a')

        for link in links:
            link_text = ['Formula', 'One', 'season', 'cars']
            if str(link.get('href')).split('_')[-4:] == link_text:
                link_season = str(WIKIPEDIA + link.get('href')).replace(' ', '')
                season = str(link.get('href')).replace(':', '_').split('_')[-5:-4][0]

                if int(season) in years:

                    req = requests.get(link_season)
                    soup = BeautifulSoup(req.content, 'html.parser')
                    team_divs = soup.findAll('div', attrs={'class': 'mw-category-group'})
                    teams_season_list = []

                    for team_div in team_divs:
                        team_links = team_div.findAll('a')
                        for team_link in team_links:
                            teams_season_list.append(team_link.get('title'))
                    seasons_chassis[season] = teams_season_list

        save_json(seasons_chassis)

        return seasons_chassis

    seasons_chassis = load_json(F1_CHASSIS)

    # Check if we got the chassis names for all seasons listed
    update_list = []

    for year in range(start_year, end_year + 1):
        if str(year) not in seasons_chassis.keys():
            update_list.append(year)

    chassis = _get_chassis_names(update_list) if update_list else seasons_chassis

    return chassis


# This function creates folders to store the train_images of each car
def create_folders(year: int, cars: list):

    if not os.path.isdir(os.path.join(TRAIN_FOLDER, str(year))):
        os.mkdir(os.path.join(TRAIN_FOLDER, str(year)))

    for car in cars:
        car = re.sub(r"[^a-zA-Z0-9]+", ' ', car)
        if not os.path.isdir(os.path.join(TRAIN_FOLDER, str(year), car)):
            os.mkdir(os.path.join(TRAIN_FOLDER, str(year), car))


def download_photos(start_year, end_year, num_photos):
    chassis = list_chassis_per_season(start_year, end_year)
    
    path = os.path.realpath('chromedriver')
    # WINDOW_SIZE = "1920,1080"
    # chrome_options = Options()
    # chrome_options.add_argument("--headless")
    # chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
    # chrome_options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(executable_path=path)
    
    for season, cars in chassis.items():
        if int(season) in range(start_year, end_year + 1):
            create_folders(season, cars)
            for car in cars:
                url = f'https://www.google.com/search?tbm=isch&q={car} F1'
                driver.get(url)
                time.sleep(0.5)
                download_list = []
                download_list.clear()

                for photo_number in range(num_photos):
                    try:
                        driver.find_element_by_xpath(
                            f'/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[1]/div[1]/div[{photo_number + 1}]/a[1]/div[1]/img').click()
                        time.sleep(0.5)
                        photo = driver.find_element_by_xpath(
                            '/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img').get_property(
                            'src')
                        if photo[-3:].lower() in ['jpg', 'png']:
                            download_list.append(photo)
                    except:
                        pass
                
                # download_list = []
                # download_list.clear()
                # for i in range(1, num_photos, 10):
                #     results = google_search(f'f1 {car}', API_KEY, CX, start=i)
                #     for item in results['items']:
                #         download_list.append(item['link'])

                if download_list:
                    car = re.sub(r"[^a-zA-Z0-9]+", ' ', car)
                    current_folder = os.path.join('train_images', season, car)

                    # Renaming current files of the folder in an organized way
                    files = os.listdir(current_folder)
                    last_id = 0
                    for number, filename in enumerate(files):
                        tmp_filename = f'sorting{filename}'
                        os.rename(os.path.join(current_folder, filename),
                                  os.path.join(current_folder, tmp_filename))

                    renamed_files = os.listdir(current_folder)
                    for number, filename in enumerate(renamed_files):

                        _, ext = os.path.splitext(filename)
                        new_filename = f'{str(number + 1000)}{ext}'
                        os.rename(os.path.join(current_folder, filename),
                                  os.path.join(current_folder, new_filename))

                    # Saving new files to the folder
                    count = len(renamed_files)
                    for link in download_list:
                        _, ext = os.path.splitext(link)
                        path = os.path.join(current_folder, f'{str(1000+count+1)}{ext}')

                        try:
                            ulib.urlretrieve(link, path)
                            print(f'Arquivo {path} salvo!')
                            count += 1
                        except Exception as e:
                            print(e)
                            print(f"Couldn't save {link}")


def google_search(search_term, api_key, cse_id, start, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, searchType='image', start=start, **kwargs).execute()
    return res
