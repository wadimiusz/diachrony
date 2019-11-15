import requests
import re
import os
from smart_open import open
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import random
import json


def getHrefs(link):
    links = []
    r = requests.get(link)
    refs = re.split('href="', re.split('<svg class="control_mini-icon control_mini-icon-prev">', re.split('">Все материалы</a></div></div></div><div id="context_3"></div></div></section><div id="root_footer">', r.text)[0])[1])
    for i in refs:
        ilink = ""
        if i.startswith("/news/"):
            ilink = re.split('">',i)[0]
        if i.startswith("/article/"):
            ilink = re.split('">',i)[0]
        if ilink:
            new_link = f"https://lenta.ru{ilink}"
            links.append(new_link)
    return links


def time_period(year):
    print("Generating links within the year")
    time_period_links = []
    for month in range(1, 12):
        if month < 10:
            month = "0" + str(month)
        for day in range(1, 32):
            if day < 10:
                day = "0" + str(day)
            onlink = "https://lenta.ru/news/" + str(year) + "/" + str(month) + "/" + str(day) + "/"
            try:
                curr_links = getHrefs(onlink)
                if curr_links:
                    time_period_links.extend(curr_links)
                else:
                    pass
            except:
                pass
    return time_period_links


def getArticleTextLenta(link):
    r = requests.get(link)
    try:
        text = re.split('<div class=', re.split('"articleBody"', r.text)[1])[0]
    except IndexError:
        return ""
    bs = BeautifulSoup(text, "lxml")
    text = bs.get_text()
    time.sleep(random.uniform(0, 3))
    return text.strip("><")


def crawl(hrefs, to_save):
    stats = {"2019": 0}
    progress_bar = tqdm(desc="Getting texts...", total=len(hrefs))
    for i, link in enumerate(hrefs):
        try:
            text = getArticleTextLenta(link)
            if text:
                filename, text_len = text.split()[0], len(text.split())
                stats["2019"] += text_len
                with open(to_save + os.sep + "{}_{}".format(filename, i) + ".txt.gz", "a", encoding="utf-8") as f:
                    f.write(text)
                    f.close()
            else:
                pass
        except:
            print(f"FAILED {link}")
            pass
        progress_bar.update(1)
    progress_bar.close()
    return stats


def main():
    hrefs = time_period(2019)
    curr_dir_path = os.getcwd()
    to_save = os.path.join(curr_dir_path, "lenta/2019")
    os.makedirs(to_save, exist_ok=True)
    get_statistics_and_crawl = crawl(hrefs, to_save)
    print(get_statistics_and_crawl)
    with open("lenta_2019_statistics.json", "w", encoding="utf-8") as c:
        json.dump(get_statistics_and_crawl, c, ensure_ascii=False, indent=4)
    print("I'm done")


if __name__ == "__main__":
    main()
