import requests
import streamlit
from bs4 import BeautifulSoup
import random
import streamlit as st


def get_random_image_url(unplash_urls):
    index = random.randint(0, len(unplash_urls))
    return unplash_urls[index]


def get_category_soup(category):
    response = requests.get(f'https://unsplash.com/t/{category}')
    category_soup = BeautifulSoup(response.text, 'html.parser')
    return category_soup


def get_image_urls_from_soup(soup):
    unsplash_urls = []
    for img in soup.find_all('img'):
        try:
            img_url = img['src']
            if 'https://images.unsplash.com/photo' in img_url:
                unsplash_urls.append(img_url)
        except:
            pass
    return unsplash_urls


def scrape_urls(categories):
    category_image_urls = {}
    for category in categories:
        category_soup = get_category_soup(category)
        category_image_urls[category] = get_image_urls_from_soup(category_soup)
    return category_image_urls
