import requests
from bs4 import BeautifulSoup
from random import randint
from time import sleep
import os
import csv
from string import punctuation

punct = punctuation + '«»—…“”*№–'


def read_page(link: str):
    user_agent = 'Chrome/72.0.3626.119'
    try:
        response = requests.get(link, timeout=100000, headers={'user-agent': user_agent}).text
        page = BeautifulSoup(response, "lxml")
        if not page.h1.text == '404':
            return page
        else:
            return None
    except requests.HTTPError:
        return None


def get_issues(startpage: str):
    link_path = "novaya_gazeta_links.txt"
    if os.path.exists(link_path):
        print("Opening links.txt")
        issues = [
            link for link in open(link_path, "r", encoding="utf-8").read().split("\n") if link]
        return issues
    issues = []
    # берем ссылку на последний в 2019 году номер архива
    issue_number = int(startpage.split('/')[-1])
    first_issue_number = 2207  # первый номер 2015 года
    while issue_number >= first_issue_number:
        issues.append('https://www.novayagazeta.ru/issues/' + str(issue_number))
        issue_number -= 1
    return issues


def get_articles(issues: list):
    # sections = ("Культура", "Общество", "Политика", "Экономика", "Спорт")
    articles = []
    issues = [i for i in issues if i.startswith("https")]
    for issue in issues:
        sleep(randint(1, 3))
        page = read_page(issue)
        print(issue)
        if page is not None:
            for i in page.find_all("h2"):
                # if i.text in sections:
                for j in i.parent.find_all("a", href=True):
                    if ("https://www.novayagazeta.ru" + j["href"]) not in articles:
                        articles.append("https://www.novayagazeta.ru" + j["href"])
                    else:
                        continue
    print('Ссылки на статьи собраны')
    return articles


def get_content(link: str):
    if "articles" not in link:
        return None
    article = read_page(link)
    print(link)  # на всякий случай выводим ссылку, чтобы в случае ошибки понять, в чем дело

    if article is not None:
        author = "No author"
        try:
            title = article.title.text
            text = article.find("div", {"id": "selection-range-available"}).text
            # текст ищется именно таким образом потому, что только так в него попадают подзаголовки
            # и персоны, реплики которых встречаются в статьях,
            # и все это встает на правильные места
            # но и немножко мусора иногда попадает: например, комментарии под фото
            unwanted_dates = '@'
            date = None
            if article.time is not None:
                date = article.time["datetime"][:10]
                if date.startswith(unwanted_dates):
                    return None
            try:
                description = article.find("p", {"class": "dBRdu"}).text
            except AttributeError:
                description = '-'
            # под заголовком обычно есть описание статьи

            words = (title + ' ' + description + ' ' + text).split()
            words = [w.strip(punct) for w in words if w]
            wordcount = len(words)

            content = [author, date, title, link, wordcount, '\n'.join([title, description, text])]
            return content

        except AttributeError:
            return None


def save_articles_and_get_paths(content: list):
    root = os.getcwd()
    year, month, day = content[1].split("-")
    filename = content[3].split("/")[-1].lstrip('0123456789-')[:100]

    # записываем в файлы сами тексты
    path1 = os.path.join(root, 'plain_text', year, month)
    os.makedirs(path1, exist_ok=True)
    with open(path1 + os.sep + filename + '.txt', 'w', encoding='UTF-8') as f:
        f.write(content[-1])
    content.remove(content[-1])  # теперь сам текст можно удалить
    path = os.path.join('.', 'plain_text', year, month, filename + '.txt')
    content.insert(0, path)  # добавляем в контент для метатаблицы путь к файлу
    content.insert(3, "Новая газета")  # и добавляем название газеты
    return content


def main():
    startpage = 'https://www.novayagazeta.ru/issues/2903'
    issues = get_issues(startpage)  # генерируем ссылки на номера газеты
    articles = get_articles(issues)  # со страниц номеров собираем ссылки на статьи

    wordcount = 0

    f = open("ng_metatable.csv", "w", encoding="UTF-8")
    csv_writer = csv.writer(f, delimiter='\t')
    csv_writer.writerow(['path', 'author', 'date', 'source', 'title', 'url', 'wordcount'])
    # запишем в csv-файл название колонок

    for article in articles:  # и будем считывать контент по одной статье
        sleep(randint(1, 3))
        content = get_content(article)  # получаем контент
        if content is not None:
            wordcount += content[-2]
            content_with_paths = save_articles_and_get_paths(content)  # записываем его в файлы
            csv_writer.writerow(content_with_paths)  # пишем в метатаблицу

            # на экран выводим количество сохраненных слов
            print('{} слов сохранены'.format(wordcount))

    f.close()


if __name__ == '__main__':
    main()
