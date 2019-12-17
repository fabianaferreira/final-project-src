from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
import requests
import csv


options = Options()
# options.add_argument('--headless')
firefox = webdriver.Firefox(options=options)
url = "http://www.acessibilidadebrasil.org.br/libras_3/"
firefox.get(url)

with open('annotations.csv', 'w') as csvfile:
	letters = firefox.find_element_by_xpath("//*[@id='filter-letter']")
	filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	# Writing title
	filewriter.writerow(['palavra', 'video_url', 'filename', 'classe_gramatical', 'assunto', 'configuracao_mao'])

	for li in letters.find_elements_by_css_selector('li'):
		print("Starting loop for letter " + li.text)
		a = firefox.find_elements_by_id("letter-" + li.text[-1:])
		a[0].click()

		firefox.implicitly_wait(10) # seconds

		words = WebDriverWait(firefox, 10).until(EC.presence_of_element_located((By.XPATH, "//*[@id='input-palavras']")))

		for word in words.find_elements_by_css_selector('option'):
			wordText = word.text
			if (wordText != "-- SELECIONE --"):
				word.click()

				filename = wordText + ".mp4"
				print("Extracting info for sign: " + wordText)
				try:
					video = WebDriverWait(firefox, 5).until(
						EC.presence_of_element_located(
							(By.XPATH, "//*[@id='videojs']"))
					)
				finally:
					source = video.find_element_by_css_selector('source')
					videoUrl = source.get_attribute('src')
					grammar = firefox.find_element_by_xpath("//*[@id='input-classe']").text
					subject = firefox.find_element_by_xpath("//*[@id='input-assunto']").text
					image = firefox.find_element_by_xpath("//*[@id='input-mao']")
					imageUrl = image.find_element_by_css_selector('img').get_attribute('src')
					filewriter.writerow([wordText, videoUrl, filename, grammar, subject, imageUrl])

					# r = requests.get(videoUrl, allow_redirects=True)
					# open(filename, 'wb').write(r.content)

firefox.quit()
