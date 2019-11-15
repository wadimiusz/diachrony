import requests
import re
import os
from smart_open import open
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import random
import json


def getArticleTextKP(link):
    r = requests.get(link)
    try:
        n_id = re.split("',", re.split("'post_id':'", r.text)[1])[0]
    except:
        n_id = "unknown_id"
    try:
        text = re.split('"articleSection":', re.split('"articleBody":', r.text)[1])[0].strip().strip(',"')
        beaux_text = BeautifulSoup(text, "lxml")
        text = beaux_text.get_text()
        time.sleep(random.uniform(1, 3))
    except:
        pass
    return n_id, text


def crawl(start, end, to_save):
    stats = {"2019": 0}
    progress_bar = tqdm(desc="Getting texts...", total=start - end)
    while True:
        link = "http://www.kp.ru/online/news/" + str(start) + "/"
        try:
            #print(f"Getting a text from {link}")
            filename, text = getArticleTextKP(link)
            text_len = len(text.split())
            stats["2019"] += text_len
            with open(to_save + os.sep + f"{filename}" + ".txt.gz", "a", encoding="utf-8") as f:
                f.write(text)
                f.close()
        except:
            print(f"FAILED {link}")
            pass
        start -= 1
        progress_bar.update(1)
        if start == end:
            progress_bar.close()
            break
    return stats


def main():
    start, end = 3672811, 3345000
    curr_dir_path = os.getcwd()
    to_save = os.path.join(curr_dir_path, "KP/2019")
    os.makedirs(to_save, exist_ok=True)
    get_statistics_and_crawl = crawl(start, end, to_save)
    print(get_statistics_and_crawl)
    with open("kp_2019_statistics.json", "a", encoding="utf-8") as c:
        json.dump(get_statistics_and_crawl, c, ensure_ascii=False, indent=4)
    print("I'm done")


if __name__ == "__main__":
    main()
