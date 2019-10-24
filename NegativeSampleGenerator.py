# This script was used to randomly generate the negative samples we used for Training and Testing.
import random
import csv

positive_samples = []
negative_samples = []


def create_negative_sample(positive_sample):
    negative_sample = ''
    gc_content = 0
    for c in positive_sample:
        if c == 'G' or c == 'C':
            gc_content = gc_content + 1
    gc_content = gc_content / len(positive_sample)
    for i, c in enumerate(positive_sample):
        random1 = random.random()
        random2 = random.random()
        if random1 < gc_content:
            if random2 < 0.5:
                negative_sample = negative_sample + 'C'
            else:
                negative_sample = negative_sample + 'G'
        else:
            if random2 < 0.5:
                negative_sample = negative_sample + 'A'
            else:
                negative_sample = negative_sample + 'T'
    return negative_sample


with open('negative_samples.csv', 'w', newline='') as writeFile:
    writer = csv.writer(writeFile)
    with open('positive_samples.csv', 'r') as readFile:
        reader = csv.reader(readFile)
        for row in reader:
            ns = create_negative_sample(row[0].strip())
            #negative_samples.append([create_negative_sample(d), '0'])
            writer.writerow([ns, 0])
writeFile.close()
