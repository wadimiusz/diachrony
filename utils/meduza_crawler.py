import os
import json
import requests
import gzip
from tqdm import tqdm
from time import sleep
from random import randint
from smart_open import open
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.parse import urlencode, urljoin
import sys


class Meduza(object):
    def __init__(self):
        self.curr_dir = os.getcwd()
        self.main_url = "https://meduza.io/"
        # self.w4 = "api/w4"
        self.w4 = "api/v3"
        self.api = "https://meduza.io/{}/search?".format(self.w4)
        self.sections = ["news", "articles", "shapito"]
        self.years = self.time_period(2020, 2021)
        self.year = 2020
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/47.0.3411.123 YaBrowser/16.2.0.2314 Safari/537.36"
        }

    @staticmethod
    def get_response(url):
        response = urlopen(url)
        headers = dict(response.headers)
        data = response.read()
        if headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
        data = data.decode("utf-8").replace("\xa0", " ")
        return json.loads(data)

    def get_article(self, url: str) -> dict:
        if not url.startswith(self.main_url):
            url = urljoin(self.main_url, url)
        if self.w4 not in url:
            url = url.replace(self.main_url, "{}{}/".format(self.main_url, self.w4))
        return self.get_response(url)["root"]

    @staticmethod
    def time_period(start, end):
        return [str(year) for year in range(start, end)]

    def filter_url(self, url, documents):
        if url == "nil":
            url = documents["nil"]["root"]["url"]
        if not any([url.startswith(section) for section in self.sections]):
            return ""
        self.year = url.split("/")[1]
        return url if self.year in self.years else ""

    def article_urls(self, url):
        urls = set()
        documents = requests.get(url, headers=self.headers).json()["collection"]
        for url in documents:
            if self.filter_url(url, documents):
                urls.add(url)
        return urls

    def section_urls(self, section, break_year="2019"):
        urls = set()
        for page in range(10000):
            sleep(randint(1, 2))
            try:
                payload = {
                    "chrono": section,
                    "page": page,
                    "per_page": 30,
                    "locale": "ru",
                }
                batch_urls = self.article_urls(self.api + urlencode(payload))
                if self.year == break_year:
                    break
                if batch_urls:
                    urls = urls.union(batch_urls)
            except TimeoutError:
                continue
        return urls

    def extract_text(self, url, types=("p", "context_p", "lead", "blockquote")):
        sleep(randint(1, 2))
        data = self.get_article(self.main_url + url)
        try:
            content = data["content"]
            if "blocks" in content:
                text = " ".join(
                    [
                        block["data"]
                        for block in content["blocks"]
                        if block["type"] in types
                    ]
                )
                text = (
                    BeautifulSoup(text, "lxml").get_text().replace("\xa0", " ").strip()
                )
                return text
            else:
                return None
        except KeyError:
            return None

    def save_text(self, url, ext=".txt.gz"):
        url_ = url.split("/")
        year, section, filename = (
            url_[1],
            url_[0],
            "_".join(url_[-1].split("-")[:3]),
        )
        to_save = os.path.join(self.curr_dir, "meduza", year)
        os.makedirs(to_save, exist_ok=True)
        file_path = os.path.join(to_save, "{}_{}.{}".format(section, filename, ext))
        if not os.path.exists(file_path):
            text = self.extract_text(url)
            if text:
                text_len = len(text.split())
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
                    f.close()
                return year, text_len
        return None, None

    def crawl_sections(self):
        statistics = {}
        for section in self.sections:
            print(section, file=sys.stderr)
            curr_section_urls = self.section_urls(section)
            progress_bar = tqdm(
                desc="Getting from section {}...".format(section),
                total=len(curr_section_urls),
            )
            for url in curr_section_urls:
                year, text_len = self.save_text(url)
                if year:
                    statistics[year] += text_len
                progress_bar.update(1)
            progress_bar.close()
        return statistics


def main():
    meduza_crawler = Meduza()
    crawl_and_count = meduza_crawler.crawl_sections()
    print(crawl_and_count)


if __name__ == "__main__":
    main()
