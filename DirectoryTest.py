import os

rootdir = '/root/Desktop/Machine_Learning/Project-SpamDetection'

for subdirs,dir,files in os.walk(rootdir):
    '''for dir in dirs:
        if(dir in ('beck-s','BG','farmer-d','GP','kaminski','kitchen-l','lokay-m','SH','williams-w3')):
            continue
        else:
            print dir'''
    for file in files:
        print os.path.join(subdirs, file)