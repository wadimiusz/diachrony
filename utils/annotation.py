import csv

with open('corpus.csv', 'r') as infile:
    reader = csv.DictReader(infile)
    data = {}
    for row in reader:
        for header, value in row.items():
            try:
                data[header].append(value)
            except KeyError:
                data[header] = [value]

output = open('results.csv', 'a')


words = list(zip(data['WORD'], data['OLD_CONTEXTS'], data['NEW_CONTEXTS']))

dict = {}

for item in words:
    print(item[0], '\n', item[1], '\n', item[2], '\n')
    answer = input(
        "Оцените, насколько изменилось значение/употребление слова от 0 (совсем не изменилось) до 3 (полностью изменилось): ")
    if answer == 'стоп':
        break
    dict[item[0]] = answer
    output.write(item[0] + ',' + answer + '\n')

output.close()

#sorry seems to be the easiest word in our case
