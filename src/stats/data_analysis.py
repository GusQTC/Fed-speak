import os

import matplotlib.pyplot as plt

# Directory paths
fomc_dir = 'fomc_minutes'
speeches_dir = 'speeches'

statements_dir = 'statements'
testimonies_dir = 'testimonies'



# Count the number of files in fomc_minutes directory
fomc_count = len(os.listdir(fomc_dir))

# Count the number of files in speeches directory
speeches_count = len(os.listdir(speeches_dir))

# Count the number of files in statements directory
statements_count = len(os.listdir(statements_dir))

# Count the number of files in testimonies directory
testimonies_count = len(os.listdir(testimonies_dir))

# Create a bar graph
labels = ['FOMC Minutes', 'Speeches', 'Statements', 'Testimonies']
counts = [fomc_count, speeches_count, statements_count, testimonies_count]
print(counts)

plt.bar(labels, counts)
plt.xlabel('File types')
plt.ylabel('Number of Files')
plt.title('File Count By Category')
plt.savefig('file_count.png')
plt.show()


