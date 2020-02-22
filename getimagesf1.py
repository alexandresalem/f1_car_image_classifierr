#Ajsutar para pegar fotos da Racing Point


from selenium import webdriver
from bs4 import BeautifulSoup as Soup
import requests as rq
from selenium.webdriver.common.keys import Keys
import os
import time
import ast
import urllib.request as ulib


chrome_path = 'chromedriver.exe'

#list = os.listdir('f1cars')
list = ['Haas']

search_list = [i.replace(' ','+')+'+Formula1+2019' for i in list]
print(list)

driver = webdriver.Chrome(chrome_path)

def download_photos(search_terms_list, photos_amount):
    for i in search_terms_list:
        download_list = []
        download_list.clear()
        url = f'https://www.google.com/search?tbm=isch&q={i}'
        driver.get(url)

        for el in range(photos_amount):
            try:
                driver.find_element_by_xpath(f'/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[1]/div[1]/div[{el+1}]/a[1]/div[1]/img').click()
                time.sleep(2)
                img = driver.find_element_by_css_selector(
                    '#Sva75c > div > div > div.pxAole > div.tvh9oe.BIB1wf > div > div.OUZ5W > div.zjoqD > div > div.v4dQwb > a > img').get_property(
                    'src')

                if img[-3:] == 'jpg':
                    download_list.append(img)

            except:
                pass
        id = search_terms_list.index(i)
        folder = list[id]
        if len(download_list) > 0:
            for item, link in enumerate(download_list):
                if not os.path.isdir(f'f1cars/{folder}'):
                    os.mkdir(f'f1cars/{folder}')


                path = os.path.join(f'f1cars/{folder}/{item + 100}.jpg')

                try:
                    ulib.urlretrieve(link, path)
                    print(f'Arquivo {path} salvo!')
                except:
                    print('Failed ULIB')




download_photos(search_list,50)



