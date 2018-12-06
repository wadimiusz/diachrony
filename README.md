# diachrony
This is the repository of the project that uses distributional semantics to track semantic shifts in words within short time spans.

The tead leam of the project in switched monthly.

The november team lead is Vadim Fomin (https://github.com/wadimiusz).

The december team lead is Julia Rodina (https://github.com/juliarodina).

**Встреча 7.11.2018**
Присутствуют все (Андрей, Даша, Вадим, Юля, Света)
- Сформулировали краткосрочные и долгосрочные задачи по проекту в целом,
- определили задачи к следущему созвону: с каким материалом необходимо ознакомиться, какие статьи прочитать,
- выбрали текущего тимлида.

_Краткосрочное:_
- завести репозиторий проекта
- ознакомиться с https://rusvectores.org/ru/about/ и прослушать лекцию в конце страницы, если нет понимания того, что там говорится
- прочитать статьи из https://rusvectores.org/news_history/References/, прежде всего https://rusvectores.org/news_history/References/diachronic_embeddings_survey.pdf и https://rusvectores.org/news_history/References/stanford_agenda_2018.pdf
- составить список корпусов
- составить список понятий для анализа
- завести ридми с описанием, инфой о том, кто тимлид и как они сменяются
- составить протокол первого созвона
- определиться с обязанностями тимлида, в каком порядке и как часто они чередуются
- научиться работать с gensim, если ещё не; особенно с помощью https://github.com/akutuzov/webvectors/blob/master/preprocessing/rusvectores_tutorial.ipynb

_Долгосрочное:_
- написать ТЗ к декабрю
- чётко сформулировать гипотезу (research question) 
- выступить дважды на семинаре
- опубликовать статью
- сделать веб-интерфейс

**Встреча 15.11.2018**
Присутствуют Андрей, Даша, Вадим (текущий тимлид) и Юля.
- Подготовка к презентации, назначенной на 17 ноября.

**Встреча 22.11.2018**
Присутствуют Андрей, Вадим (текущий тимлид), Юля, Даша

_Обсуждалось:_ <br>
Как прошла презентация и что сделать дальше

_Задачи:_ 
- Перечитать статьи
- Выбрать слова
- Составить техзадание

_Что должно быть в техзадании?_
- Используемые датасеты
- Используемые методы
- Используемые слова
- Используемый метод эвалюейшн

_Что конкретно сделать?_ <br>
**Методы:** всем прочитать обзорные статьи, раскидать 5-6 методов новых лет по людям и в следующий раз представить их, решить и выбрать.<br>
**Слова:** взять простую модель и простой датасет, составить черновой список слов. Это должно стать результатом коворкинг-сешна, который надо назначить.<br>
**Оргмомент:** карточки должны обзавестись ответственными людьми и дедлайнами.

**Встреча 29.11.2018**
Присутствуют Андрей, Вадим (текущий тимлид), Юля, Даша<br>
Обсуждение выступления на следующем НИСе, выступления перед друг другом с презетациями статей и коворкинг-сешна

**Co-working session 05.12** Вадим, Даша, Юля (текущий тимлид)<br>

Взяли две модели с русвекторес, сравнили их с помощью библиотеки gensim. Код в файле “naive_comparison.py”.  
*Алгоритм*: берём 1000 самых частотных прилагательных, 10 ближайших соседей каждого прилагательного в той и другой модели жаккаровой мерой, потом сортируем и выводим самые отличающиеся слова.  
*Результаты*: есть много проблем с данными, грязные данные помешали остановиться на каких-то конкретных словах. 

**Встреча 06.12.2018** Андрей, Вадим, Юля (текущий тимлид)<br>
Обсудили результаты коворкинга. Сделать еще помимо Жаккарда Kendall tau distance, использовать для сравнения корпуса отсюда: https://rusvectores.org/news_history/models/ 2 смежных года, но можно и больший промежуток, создать предварительный список прилагательных, сгенерированный двумя методами. Попробовать Прокрустово выравнивание.


