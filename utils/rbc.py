import re
import os
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
from string import punctuation


def get_sitemap(sitemaps):
    request = requests.get(sitemaps)
    sitemaps = [sitemap.get_text() for sitemap in bs(request.text, 'lxml').find_all('loc')]
    print('{} sitemaps crawled'.format(len(sitemaps)))
    return sitemaps


def crawl_sitemap(sitemaps, n=512):
    '''для n мэпов, где n=512 - max кол-во мэпов рбк - создаем большой словарь ссылок на тексты
    и дат их создания такой, что key:value = link:date и len(all_links) == len(set(all_links)) == len(all_dates),
    т.е. ссылки корректно мэтчатся с датами'''
    user_agent = 'Chrome/72.0.3626.119'
    all_links, all_dates = [], []
    if n == 0 or n > 512:
        raise Exception('N should not exceed 512 or be of 0. The entered value of N was: {}.'.format(n))
    for sitemap in get_sitemap(sitemaps)[n:]:
        sitemap = bs(requests.get(sitemap, timeout=100000, headers={'user-agent': user_agent}).text, 'lxml')
        chunks = re.findall('http.+\s.+T', str(sitemap))
        # http://top.rbc.ru/economics/04/10/2011/618554.shtml</loc>\n<lastmod>2012-02-06T
        text_chunks = [chunk.replace('</loc>\n<lastmod>', '*').replace('T', '') for chunk in chunks if 'rbc' in chunk \
                       and 'wmv' not in chunk and 'video' not in chunk]
        all_links.extend([text_chunk.split('*')[0] for text_chunk in text_chunks])
        all_dates.extend([text_chunk.split('*')[1] for text_chunk in text_chunks])
    print('{} links crawled in total'.format(len(all_links)))
    return dict(zip(all_links, all_dates))


def get_dates(rbc):
    '''ловим ссылки в указанном диапазоне времени'''
    clean_dates, clean_rbc = [], {}
    for date in rbc.values():
        clean_dates.extend(re.findall('201[5-9]-\d{2}-\d{2}', date))
        #clean_dates.extend(re.findall('2015-0[7-9]-\d{2}', date))
        #clean_dates.extend(re.findall('2015-1[0-2]-\d{2}', date))
    for url, date in rbc.items():
        if date in clean_dates:
            clean_rbc[url] = [date]
    print('{} links crawled;'.format(len(clean_rbc.keys())))
    return clean_rbc


def retrieve(site):
    '''в except попадают ссылки, при переходе на которых нет текста, кроме заголовков,
    уже встречавшихся в словаре. изначально хотелось удалить такие ссылки из словаря,
    добавив их в exceptions, но во время итерации это сделать нельзя. потом я устал
    и пришел к выводу, что пустые текстовые файлы в функции make_dirs_and_files по таким
    ссылкам создаваться не будут, но сами ссылки и их соответствующие значения будут храниться
    в датафрейме'''
    rbc = get_dates(site)
    # exceptions = []
    for url in rbc.keys():
        print("Processing url: {}".format(url))
        try:
            page = requests.get(url)
            soup = bs(page.text, 'html5lib')
            try:
                # title [1]
                rbc[url].extend([p.get_text().split(' :: ')[0] for p in soup.select('title')])
                # source [2]
                rbc[url].extend([p.get_text().split(' :: ')[2] for p in soup.select('title')])
                # text [3]
                text = bs(' '.join([re.sub('[ ]{2,}', '', p.text) for p in soup.find_all('p')]), \
                'html5lib').text.replace('\xa0', ' ').replace('\n', '').replace('\u200b', '')
                rbc[url].append(text)
                # author [4]
                author = ""
                rbc[url].extend([author] if author else ['unknown'])
                # wordcount [5]
                rbc[url].append(len([w.strip(punctuation + '«»—…“”*№– ') for w in \
                              text.split()]))
            except:
                print('Cannot open', str(url))
            # exceptions.append(url)
                rbc[url].extend(['-', '-', '-', '-', '-'])
        except:
            rbc[url].extend("- - - - -".split())
    return rbc # exceptions


def make_dirs_and_files(rbc):
    '''создаем пустой датафрейм'''
    rbc_df = pd.DataFrame(columns=['path', 'author', 'date', 'source', 'title', 'url', 'wordcount'])
    for link in rbc.keys():
        '''название файлов'''
        url = rbc[link]
        try:
            name = url[4].replace(' ', '_').replace(',', '') + '_' + url[0].replace('-', '_')
        except:
            name = 'unknown' + '_' + url[0].replace('-', '_')
        '''строим дерево'''
        year, month = url[0].split('-')[0], url[0].split('-')[1]
        plain_path = os.path.join(os.getcwd(), 'plain_texts', '{}'.format(year), '{}'.format(month))
        os.makedirs(plain_path, exist_ok=True)
        file_path = os.path.join(plain_path, name + '.txt')
        '''кормим датафрейм'''
        rbc_df.loc[len(rbc_df)] = [file_path, url[4], url[0], url[2], \
                                       url[1], link, url[5]]
        '''сохраняем файлы в соответствующих папках так, что если за текущий день есть другая
            статья от этого автора или автор неизвестен, склеиваем тексты статей; при этом
            тексты длины 0 из exceptions не записываются в файлы'''
        if len(url[3]) > 1:
            with open(plain_path + os.sep + name + '.txt', 'a', encoding='utf-8') as f:
                f.write(url[3])
                f.close()
            print('Plain {} created'.format(name + '.txt'))
    return rbc_df.to_csv('rbc.csv', sep='\t', encoding='utf-8', index=False)


def main():
    rbc_sitemap = 'https://www.rbc.ru/sitemap_index.xml'
    sitemap = crawl_sitemap(rbc_sitemap, n=512)
    rbc = retrieve(sitemap)
    make_dirs_and_files(rbc)


if __name__ == '__main__':
    main()
