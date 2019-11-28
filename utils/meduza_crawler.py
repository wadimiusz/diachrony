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


class Meduza(object):
    def __init__(self):
        self.curr_dir = os.getcwd()
        self.main_url = "https://meduza.io/"
        self.w4 = "api/w4"
        self.api = "https://meduza.io/{}/search?".format(self.w4)
        self.sections = ["news", "articles", "shapito", "razbor", "feature"]
        self.years = self.time_period(2015, 2020)
        self.year = None
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) "
                                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                                      "Chrome/47.0.3411.123 YaBrowser/16.2.0.2314 Safari/537.36"}

    def get_article(self, url: str) -> dict:
        if not url.startswith(self.main_url):
            url = urljoin(self.main_url, url)
        if self.w4 not in url:
            url = url.replace(self.main_url, "{}/{}".format(self.main_url, self.w4))
        return self.get_response(url)["root"]    
    
    @staticmethod
    def get_response(url):
        response = urlopen(url)
        headers = dict(response.headers)
        data = response.read()
        # if the data is compressed using gzip, then decompress this.
        # (u is a gzip file if the first two bytes are '0x1f' and '0x8b')
        if headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
        # remove all non-breaking spaces
        data = data.decode("utf-8").replace("\xa0", " ")
        return json.loads(data)

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

    def section_urls(self, section, break_year="2014"):
        urls = set()
        for page in range(10000):
            sleep(randint(1, 3))
            try:
                payload = {
                    "chrono": section,
                    "page": page,
                    "per_page": 45,
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
        sleep(randint(1, 3))
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
                text = BeautifulSoup(text, "lxml").get_text().replace("\xa0", " ").strip()
                return text
            else:
                return None
        except KeyError:
            return None

    def save_text(self, url, ext=".txt.gz"):
        text = self.extract_text(url)
        if text:
            url_ = url.split("/")
            year, section, filename = (
                url_[1],
                url_[0],
                "_".join(url_[-1].split("-")[:3]),
            )
            to_save = os.path.join(self.curr_dir, "meduza", year)
            os.makedirs(to_save, exist_ok=True)
            file_path = os.path.join(to_save, "{}_{}.{}".format(section, filename, ext))
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
                f.close()
            text_len = len(text.split())
            return year, text_len
        else:
            pass

    @staticmethod
    def section_url_cache(section, section_urls, url_cache_path):
        with open(url_cache_path, "a", encoding="utf-8") as f:
            json.dump({section: list(section_urls)}, f, ensure_ascii=False, indent=4)

    def read_url_cache(self, section, url_cache_path):
        section_urls = []
        if os.path.exists(os.path.join(self.curr_dir, url_cache_path)):
            with open(url_cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
                if section in cache:
                    section_urls = set(cache[section])
        return section_urls

    def crawl_sections(self, url_cache_path="section_url_cache.json.gz"):
        statistics = {"201{}".format(i + 5): 0 for i in range(5)}
        global_urls = set()
        for section in self.sections:
            curr_section_urls = self.read_url_cache(section, url_cache_path)
            if not curr_section_urls:
                curr_section_urls = self.section_urls(section)
                self.section_url_cache(section, curr_section_urls, url_cache_path)
            progress_bar = tqdm(
                desc="Getting urls from section {}...".format(section),
                total=len(curr_section_urls),
            )
            for url in curr_section_urls:
                if url not in global_urls:
                    year, text_len = self.save_text(url)
                    statistics[year] += text_len
                progress_bar.update(1)
            global_urls.union(curr_section_urls)
            progress_bar.close()
        return statistics


def main():
    meduza_crawler = Meduza()
    crawl_and_count = meduza_crawler.crawl_sections()
    print(crawl_and_count)


if __name__ == "__main__":
    main()
