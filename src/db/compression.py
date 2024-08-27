import zlib
import base64
import os


dir = 'fomc_minutes'

if not os.path.exists(f'compressed_{dir}'):
    os.makedirs(f'compressed_{dir}')

for filename in os.listdir(dir):

    if filename.endswith('.txt'):
        with open(dir + '/' + filename, 'r', encoding='utf-8') as file_reader:
            text = file_reader.read()
            compressed = base64.b64encode(zlib.compress(text.encode())).decode()
            
            with open(f'compressed_{dir}' + filename, 'w', encoding='utf-8') as file_writer:
                file_writer.write(compressed)
                print('Compressed Text:', filename)
        os.remove(dir +'/' + filename)







#decompressed = zlib.decompress(base64.b64decode(text.encode())).decode()
#print('Decompressed Text:', decompressed)