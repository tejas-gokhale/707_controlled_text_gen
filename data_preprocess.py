from numpy import genfromtxt
import csv
import numpy as np
# summaries = genfromtxt('./MovieSummaries/plot_summaries_sorted.csv', delimiter='\t')
# # metadata = genfromtxt('./MovieSummaries/movie.metadata.tsv', delimiter='\t', usecols = (0, 8))



list_of_summ_id = []
summaries = {}
with open('./MovieSummaries/plot_summaries.csv', 'r', encoding='Latin-1') as f:
	f_reader = csv.reader(f)
	i = 0
	for row in f_reader:
		# i = i+1
		# print(i)
		# print(row[0])
		list_of_summ_id.append(int(row[0]))
		summaries[int(row[0])] = row[1]


print('done')

# print(summaries[330])
with open('./MovieSummaries/genre.csv', 'r', encoding='Latin-1') as f:
	f_reader = csv.reader(f)
	i = 0
	for row in f_reader:
		if int(row[0]) in list_of_summ_id:
			# print(row)
			row.append(summaries[int(row[0])])
			row[1], row[2] = row[2], row[1]

			# print('\t'.join(row))
			with open('./MovieSummaries/summaries_genre.txt', 'a', encoding='Latin-1') as file:
				file.write('\t'.join(row))
				file.write('\n')
				i = i+1
				if i%200 == 0:
					print(i)



# with open('./MovieSummaries/genre.csv') as f:
# 	f_reader = csv.reader(f)
# 	for row in f_reader:
# 		print(row)
# 		break