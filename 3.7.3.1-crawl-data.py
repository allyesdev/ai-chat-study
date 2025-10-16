from selenium import webdriver
from bs4 import BeautifulSoup
import csv
import time
import os

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# 아래 링크에서 버전에 맞는 크롬 드라이버 다운로드
# https://googlechromelabs.github.io/chrome-for-testing/

# mac의 경우 brew install chromedriver 로 설치
# brew install chromedriver

driver = webdriver.Chrome()

url = "https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=%EC%98%81%ED%99%94+%EB%B3%B4%EC%8A%A4+%ED%8F%89%EC%A0%90&ackey=d8kdncox"
driver.get(url)


html = driver.page_source
soup = BeautifulSoup(html, "html.parser")

SCROLL_INTERVAL = 10


def scroll_to_bottom():
    element = driver.find_element(By.CLASS_NAME, "lego_review_list")
    driver.execute_script("arguments[0].scrollBy(0,1000);", element)


with open("data/review/reviews.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text_review", "rate_review"])

    scroll_count = 0
    while scroll_count < SCROLL_INTERVAL:
        scroll_to_bottom()
        time.sleep(1)
        scroll_count += 1

    soup = BeautifulSoup(driver.page_source, "html.parser")

    review_list = soup.select(".lego_review_list ul li")
    for review in review_list:
        text_review = review.select_one(".desc").text.strip()
        rate_element = review.select_one(".lego_movie_pure_star .area_text_box")
        rate_review = rate_element.find_all(string=True)[-1].strip()
        writer.writerow([text_review, rate_review])

driver.quit()
