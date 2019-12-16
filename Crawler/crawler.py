from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests

firefox = webdriver.Firefox()
url = "http://www.acessibilidadebrasil.org.br/libras_3/"
firefox.get(url)

letters = firefox.find_element_by_xpath("//*[@id='filter-letter']")

for li in letters.find_elements_by_css_selector('li'):
    li.click()
    words = firefox.find_element_by_xpath("//*[@id='input-palavras']")
    for word in words.find_elements_by_css_selector('option'):
        wordText = word.text
        if (wordText != "-- SELECIONE --"):
            word.click()
            filename = wordText + ".mp4"
            try:
                video = WebDriverWait(firefox, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//*[@id='videojs']"))
                )
            finally:
                source = video.find_element_by_css_selector('source')
                videoUrl = source.get_attribute('src')
                r = requests.get(videoUrl, allow_redirects=True)
                open(filename, 'wb').write(r.content)
        
firefox.quit()