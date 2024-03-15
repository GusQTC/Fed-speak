import os

import matplotlib.pyplot as plt

# Directory paths
fomc_dir = 'fomc_minutes'
speeches_dir = 'speeches'



# Count the number of files in fomc_minutes directory
fomc_count = len(os.listdir(fomc_dir))

# Count the number of files in speeches directory
speeches_count = len(os.listdir(speeches_dir))

# Create a bar graph
labels = ['fomc_minutes', 'speeches']
counts = [fomc_count, speeches_count]
print(counts)

plt.bar(labels, counts)
plt.xlabel('Directories')
plt.ylabel('Number of Files')
plt.title('File Count in Directories')
plt.savefig('file_count.png')
plt.show()