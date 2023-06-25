from scapy.all import *
from tqdm import tqdm
import json

def get_str():
    nowpath = os.getcwd() + r'/'[0]
    print('nowpath: ', nowpath)
    folders = ['pre_exp']

    for i in range(len(folders)):
        filenames = os.listdir(nowpath + folders[i])
        foldersinfo = []
        print(i + 1, end=' ')
        print(len(filenames))
        for each in filenames:
            filename = nowpath + folders[i] + r'/'[0] + each
            tmp = each.split('.p')[0]
            with open(nowpath + folders[i] + r'/'[0] + tmp + '.txt', 'w') as f:
                try:
                    pr = PcapReader(filename)
                except:
                    print('file ', filename, '  error')
                    continue
                pkt = 1
                while pkt:
                    try:
                        pkt = pr.read_packet()
                        if 'Raw' in pkt:
                            f.write(str(repr(pkt)) + '\n')
                    except EOFError:
                        break

        with open(nowpath + folders[i] + r'/'[0] + tmp+'.txt', 'a') as f:
            f.write(str(foldersinfo))

def triplet_tokens():
    adrs = ['pre_exp/']
    datasets = ['normal','abnormal']

    for i in range(len(adrs)):
        if i == 0:
            clss1 = {'normal': 0, 'abnormal': 1}

        for j in range(len(datasets)):
            d = datasets[j]
            adr = adrs[i]+d+'.txt'
            f = open(adr)
            lines = [line.strip() for line in f.readlines()]
            # just a sample
            lines = lines[:1000]
            data = {}
            for line in tqdm(lines):
                text = re.split(r'\\| ',line)
                for k in range(len(text)):
                    if 'src' in text[k] or 'dst' in text[k] or 'port' in text[k]:
                        text[k]=''
                label1 = clss1[d]
                if i == 0:
                    tmp = {"text": list(filter(None, text)),
                       "label": {0: clss1[d], 1: None, 2: None, 3: None, 4: None, 5: None}}
                #keep consistent to the structure of the training sets, although we donot use the positive and negative part in the test.
                tr = {"anchor": tmp, "positive": tmp, "negative": tmp}
                tr = json.dumps(tr, ensure_ascii=False)
                if str(label1) not in data:
                    data[str(label1)] = [tr]
                else:
                    data[str(label1)].append(tr)

            w = open(adrs[i]+'test.txt', 'a')
            for tmp in data:
               w.write('\n'.join(str(v) for v in data[tmp]))

# 1. from pcap to str
get_str()

# 2. split into tokens and produce 'text.txt'
triplet_tokens()

# 3. test in the pretrain
# put the text.txt into the model
