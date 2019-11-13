# python3
# coding: utf-8

import datetime
import os
import re
import urllib.request
from bs4 import BeautifulSoup
from smart_open import open

URL = 'https://nplus1.ru'  # источник текстов


def download_page(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        page = response.read()
    return page


def write_in_file(filename, text):
    with open(filename, 'w+', encoding='UTF-8') as f:
        f.write(text)
        f.close()


def get_links_from_daily_page(date_string):
    path = "/news/" + date_string
    daily_url = URL + path
    daily_news_page = download_page(daily_url)
    soup = BeautifulSoup(daily_news_page, features="lxml")

    all_anchor_nodes = soup.find(id="main").find_all('a', href=re.compile(path))

    all_links = list(set(map(lambda node: node.get('href'), all_anchor_nodes)))

    return all_links


def parse_article_page(article_path):
    article_page = download_page(URL + article_path)
    article_html = BeautifulSoup(article_page, features="lxml")
    paragraphs = article_html.find(class_="body").findChildren('p', recursive=False)
    author = None
    if len(paragraphs) > 0 and paragraphs[-1].i:
        author = paragraphs[-1].i.extract().text
    article_text = ' '.join([p.text for p in paragraphs])
    title = article_html.h1.text
    source = URL + article_path
    wordcount = len(article_text.split(' '))
    [date] = article_html.time.get_attribute_list('content')
    article_metadata = [article_path, author, date, URL, title, source, wordcount]
    return {
        'article_text': article_text,
        'article_metadata': article_metadata
    }


def daterange(start_date, end_date):
    array = []
    for n in range(int((end_date - start_date).days + 1)):
        array.append(start_date + datetime.timedelta(n))
    return array


def main():
    start_date = datetime.date(2019, 1, 1)
    end_date = datetime.date(2019, 11, 12)
    dates = daterange(start_date, end_date)
    metadatas = [['path', 'author', 'date', 'source', 'title', 'url', 'wordcount']]
    for date in dates:
        [year, month, day] = str(date).split('-')
        path = year + os.sep + month
        corpus_path = 'n_plus_one' + os.sep + path
        if not (os.path.exists(corpus_path)):
            os.makedirs(corpus_path)
        date_string = '/'.join([year, month, day])
        article_paths = get_links_from_daily_page(date_string)
        for article_path in article_paths:
            data = parse_article_page(article_path)
            article_name = article_path.split('/')[-1]
            article_text = data['article_text']
            write_in_file(corpus_path + os.sep + article_name + '.txt.gz', article_text)
            print(f"{article_name}.txt saved")
            metadatas.append(data['article_metadata'])
    write_in_file('metadata.csv.gz', '\n'.join(['\t'.join([str(el) for el in metadata])
                                                for metadata in metadatas]))
    return


if __name__ == '__main__':
    main()
