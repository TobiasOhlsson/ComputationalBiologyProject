import csv

datasets = ['REST', 'PHF8', 'BCL3', 'ATF3', 'SIX5', 'NR3C1', 'RFX5', 'RNF2', 'CTCF1']
filter_sizes = [490, 500, 536, 480, 456, 280, 376, 404, 216]


def create_pos_csv(s, filtered):
    filter_size = filter_sizes[s]
    dataset = datasets[s]
    file = open("hgTables/" + dataset + "_hgTables.txt", "r")
    print("File opened")
    data = []
    file_name = 'samples/' + dataset + '_positive_samples.csv'
    if filtered:
        file_name = 'filtered_samples/' + dataset + '_filtered_positive_samples.csv'
    with open(file_name, 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        d = ''
        for i, line in enumerate(file):
            if line.startswith(">"):
                if i != 0:
                    d = d.replace('\n', '')
                    if not filtered:
                        writer.writerow([d, 1])
                        data.append(d)
                    elif (s > 4) and (abs(len(d) - filter_size) < 50):
                        writer.writerow([d, 1])
                        data.append(d)
                    elif (s <= 4) and (450 < len(d) < 550):
                        writer.writerow([d, 1])
                        data.append(d)
                    d = ''
            else:
                d = d + line
    print(len(data))
    writeFile.close()


for s in range(len(datasets)):
    create_pos_csv(s, True)

