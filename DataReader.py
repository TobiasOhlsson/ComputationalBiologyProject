import csv

file = open("hgTables.txt", "r")
print("File opened")
data = []
with open('positive_samples.csv', 'w', newline='') as writeFile:
    writer = csv.writer(writeFile)
    d = ''
    for i, line in enumerate(file):
        if line.startswith(">"):
            if i != 0:
                d = d.replace('\n', '')
                writer.writerow([d, 1])
                data.append(d)
                d = ''
        else:
            d = d + line
print(len(data))
writeFile.close()

