# This script was used to randomly generate the negative samples we used for Training and Testing.
import random
import csv

datasets = ['REST', 'PHF8', 'BCL3', 'ATF3', 'SIX5', 'NR3C1', 'RFX5', 'RNF2', 'CTCF1']


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


def create_neg_csv(s, filtered):
    positive_samples = []
    negative_samples = []

    dataset = datasets[s]
    positive_file_name = 'samples/' + dataset + '_positive_samples.csv'
    negative_file_name = 'samples/' + dataset + '_negative_samples.csv'
    if filtered:
        positive_file_name = 'filtered_samples/' + dataset + '_filtered_positive_samples.csv'
        negative_file_name = 'filtered_samples/' + dataset + '_filtered_negative_samples.csv'
    with open(negative_file_name, 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        with open(positive_file_name, 'r') as readFile:
            reader = csv.reader(readFile)
            for row in reader:
                ns = create_negative_sample(row[0].strip())
                # negative_samples.append([create_negative_sample(d), '0'])
                writer.writerow([ns, 0])
    writeFile.close()


for s in range(len(datasets)):
    create_neg_csv(s, True)



