import sys
import subprocess
import requests
import asyncio
import random
import time
from bs4 import BeautifulSoup
import pandas as pd
import aiohttp
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Vérifier que les paquets nécessaires sont installés
required_packages = ['requests', 'beautifulsoup4', 'pandas', 'aiohttp', 'selenium']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Vérifier la version de Python
if sys.version_info < (3, 7):
    print("Ce script nécessite Python 3.7 ou supérieur.")
    sys.exit(1)

# URL de la page à extraire
url = "https://www.moniteurlive.com/l/"

# Utiliser une approche asynchrone pour récupérer les données de manière plus rapide et efficace.
async def fetch_data(session, url, data):
    async with session.post(url, data=data, headers={'User-Agent': 'Mozilla/5.0'}) as response:
        return await response.text()

# Utiliser les requêtes POST plutôt que les requêtes GET pour obtenir les résultats de recherche et les résultats de chaque page de produits.
# Ajouter des en-têtes de requête HTTP pour éviter d'être bloqué par le site web.
async def get_page_data(session, page_number, driver):
    data = {
        'pageNumber': str(page_number),
        'criterion': 'lotsOrder',
        'orderBy': 'desc',
        'maxPerPage': '12',
        'maxItems': '0',
        'firstIndex': '0',
        'category[]': '28'
    }
    response_text = await fetch_data(session, url, data)
    soup = BeautifulSoup(response_text, 'html.parser')
    product_elements = soup.find_all('div', {'class': 'product-detail-wrapper'})
    data = []
    for product_element in product_elements:
        title = product_element.find('div', {'class': 'product-title'}).text.strip()
        description = product_element.find('div', {'class': 'product-description'}).text.strip()
        price = product_element.find('div', {'class': 'price'}).text.strip()
        estimation = product_element.find('div', {'class': 'estimation'}).text.strip()
        data.append({'Titre': title, 'Description': description, 'Prix': price, 'Estimation': estimation})
    return data

# Parcourir chaque page de la catégorie High Tech, Multimédia et Informatique pour extraire les informations des produits
async def scrape_pages():
    # Initialiser la variable soup
    async with aiohttp.ClientSession() as session:
        driver = webdriver.Chrome()
        driver.get(url)
        # Attendre que la page se charge complètement
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.product-detail-wrapper')))
        response_text = driver.page_source
        soup = BeautifulSoup(response_text, 'html.parser')
        last_page_link = soup.find('a', {'title': 'dernière page'})
    if last_page_link:
        page_count = int(last_page_link.text)
    async with aiohttp.ClientSession() as session:
        driver = webdriver.Chrome()
        driver.get(url)
        # Attendre que la page se charge complètement
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.product-detail-wrapper')))
        response_text = driver.page_source
        soup = BeautifulSoup(response_text, 'html.parser')
        last_page_link = soup.find('a', {'title': 'dernière page'})
    if last_page_link:
        page_count = int(last_page_link.text)
        
    # Extraire les données de chaque page à l'aide de la fonction get_page_data
    tasks = []
    async with aiohttp.ClientSession() as session:
        driver = webdriver.Chrome()
        driver.get(url)
        # Attendre que la page se charge complètement
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.product-detail-wrapper')))
        for page_number in range(1, page_count + 1):
            task = asyncio.ensure_future(get_page_data(session, page_number, driver))
            tasks.append(task)
        data = []
        for result in await asyncio.gather(*tasks):
            data.extend(result)
    driver.quit()

    # Afficher les données dans un DataFrame Pandas
    df = pd.DataFrame(data)
    print(df)