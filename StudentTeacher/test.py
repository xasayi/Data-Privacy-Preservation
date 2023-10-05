# SANITY CHECK TESTING FILE
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '/Users/sarinaxi/Desktop/Thesis')

def get_data_downsamples(filename):
    with open(filename) as f:
        content = f.read()
    lines = content.split('\n')
    lines = np.array([i.split('\t') for i in lines][:-2])
    
    df = pd.DataFrame(lines, columns=['type', 'sms'])
    ham = df[df['type']=='ham'].drop_duplicates()
    spam = df[df['type']=='spam'].drop_duplicates()
    ham_less = ham.sample(n = len(spam), random_state = 44)
    df = pd.concat([ham_less, spam]).reset_index().drop(columns=['index'])
    idx = np.random.permutation(df.index)
    ret = df.reindex(idx)
    dic = {'data':list(ret['sms']), 'label': [1 if i == 'spam' else 0 for i in ret['type']]}
    return dic

if __name__ == '__main__':
    filename = '/Users/sarinaxi/Desktop/Thesis/SpamDetector/data/smsSpam/SMSSpamCollection.txt'
    dic = get_data_downsamples(filename)