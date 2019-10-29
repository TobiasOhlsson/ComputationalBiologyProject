import csv

set = 0
filtered = True
filter_sizes = [216, 490, 500, 536, 480, 280, 456, 376, 404]
filter_size = filter_sizes[set]

datasets = ['CTCF1', 'REST', 'PHF8', 'BCL3', 'ATF3', 'NR3C1', 'SIX5', 'RFX5', 'RNF2']
dataset = datasets[set]
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
                if (not filtered) or (len(d) == filter_size):
                    writer.writerow([d, 1])
                    data.append(d)
                d = ''
        else:
            d = d + line
print(len(data))
writeFile.close()

