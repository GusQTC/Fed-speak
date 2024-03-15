import os

files = os.listdir('speeches')

for file in files:
    print(file)
    new_name = file.replace('htm.txt', 'txt')
    print(new_name)
    os.rename(f'speeches/{file}', f'speeches/{new_name}')
