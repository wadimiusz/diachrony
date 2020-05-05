import urllib.request
import re
import os
from tqdm import tqdm
import json
from smart_open import open
from bs4 import BeautifulSoup


articles = []
global_url = r'http://www.the-village.ru/news?&page='


curr_dir_path = os.getcwd()
to_save = os.path.join(curr_dir_path, 'the_village/2020')
if not os.path.exists(to_save):
    os.makedirs(to_save)

total = 0


def make_request(url):
    response = urllib.request.urlopen(url)
    soup = BeautifulSoup(response, 'html.parser')
    return soup


def find_dates(global_url):
    page_range = []
    for i in range(1, 25):
        print('Номер страницы: ' + str(i))
        url = global_url + str(i)
        soup = make_request(url)
        for item in soup.find_all('h2'):
            if '3 мая' in item:
                page_range.append(i)
                for div in str(soup.find_all('div')).split('\n'):
                    if 'p-news' in div:
                        slice = div[div.index('3 мая'):]
                        find_links(slice)
            if '1 января' in item:
                page_range.append(i)
                for div in str(soup.find_all('div')).split('\n'):
                    find_links(div)
                    break
        if page_range:
            if i > page_range[0]:
                find_links(soup.find_all('a'))
            if len(page_range) > 1:
                break


def find_links(all_a):
    link = re.findall(r"(/village/[a-z]+(-[a-z]+)*/[a-z]+-?[a-z]+/(\d+)(-\d)*(-[a-z]+)*)", str(all_a))
    for l in link:
        articles.append(l[0])


def parse_page(idx, article):
    global total
    soup = make_request(article)
    paragraphs = [p.text.strip('\t\r\n ').replace(u'\xa0', u' ') for p in soup.find_all('p')]
    for num, item in enumerate(paragraphs):
        if item == 'telegram':
            paragraphs = paragraphs[:num]
        if '© 2020 The Village.' in item:
            paragraphs = paragraphs[:num]
    text = ' '.join(paragraphs)
    wordcount = len(text.split(' '))
    total += wordcount
    title = soup.find('h1', attrs={'class': 'article-title'}).text
    filename = title if title else text.split()[0] + "_" + idx
    with open(to_save + os.sep + filename + ".txt.gz", "a", encoding="utf-8") as file:
        file.write(text)
        file.close()


def main():
    global articles
    global global_url
    global to_save
    main_url = r'http://www.the-village.ru'
    find_dates(global_url)
    articles = list(set(articles))
    progress_bar = tqdm(desc="Getting texts...", total=len(articles))
    for i, link in enumerate(articles):
        try:
            article = main_url + link
            parse_page(i, article)
        except:
            pass
        progress_bar.update(1)
    progress_bar.close()
    print("Token statistics:", total)
    with open("the_village_2020_statistics.json", "w", encoding="utf-8") as f:
        json.dump({"2020": total}, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
