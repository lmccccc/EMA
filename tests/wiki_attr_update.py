import json


source_file = "/mnt/data/mocheng/dataset/navix_dataset/wiki_15.4M/fvecs/wiki_15.4M_attr.json"

with open(source_file, 'r') as f:
    data = json.load(f)

print("example data:", data[0:5])

print("example data[1]: ", data[1])

print("len of attr[1]: ", len(data[1]))



updated_data = []

min_date = 99999999
max_date = -1

is_date = 0
for idx, record in enumerate(data):
    try:
        if len(record) == 1:
            is_date += 1
            updated_data.append([record[0]])
            if record[0][0] < min_date:
                min_date = record[0][0]
            if record[0][0] > max_date:
                max_date = record[0][0]
        else:
            updated_data.append([[0]])
    except:
        print("error record:", record, " at index:", idx)
        exit(-1)

print("is date:", is_date, " pct: ", is_date / len(data))
print("min date:", min_date, " max date:", max_date)
print("example:", updated_data[0:20])

target_file = "/mnt/data/mocheng/dataset/navix_dataset/wiki_15.4M/fvecs/wiki_15.4M_birthdate.json"

with open(target_file, 'w') as f:
    json.dump(updated_data, f)
print("updated data saved to:", target_file)