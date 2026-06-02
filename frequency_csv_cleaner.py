import csv
with open('/var/lib/postgresql/18/main/pgtoken_codebooks/cl100k_base.csv') as fin, \
     open('/var/lib/postgresql/18/main/pgtoken_codebooks/cl100k_base_clean.csv', 'w') as fout:
    reader = csv.reader(fin)
    next(reader)  # skip header
    fout.write('rank,token_id\n')
    count = 0
    for row in reader:
        fout.write(f'{row[0]},{row[1]}\n')
        count += 1
print(f'Written {count} rows')
