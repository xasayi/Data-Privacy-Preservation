from evaluate import load
import random
import numpy as np
import pickle
import time
import sys
sys.path.insert(0, '/Users/sarinaxi/Desktop/Thesis')

from SpamDetector.process_data import get_df

def create_private_public(filename, index):
    df = get_df(filename, index)
    spam = np.array([[k,1] for k, v in zip(df['data'], df['label']) if v == 1])
    ham = np.array([[k,0] for k, v in zip(df['data'], df['label']) if v == 0])

    spam_half = random.sample(range(0, len(spam)), int(len(spam)//2))
    ham_half = random.sample(range(0, len(ham)), int(len(ham)//2))

    spam_other = np.setdiff1d(np.arange(0, len(spam)-1), spam_half)
    ham_other = np.setdiff1d(np.arange(0, len(ham)-1), ham_half)
    private = np.concatenate((spam[spam_half], ham[ham_half]))
    public = np.concatenate((spam[spam_other], ham[ham_other]))

    np.random.shuffle(private)
    np.random.shuffle(public)
    print(len(private), len(public))
    return private[:10], public

def similarity(reference, prediction):
    bertscore = load("bertscore")
    references = reference*len(prediction)
    results = bertscore.compute(predictions=prediction, references=references, lang="en", model_type="distilbert-base-uncased")
    return results

if __name__ == '__main__':
    filename = '/Users/sarinaxi/Desktop/Thesis/SpamDetector/data/smsSpam/SMSSpamCollection.txt'
    private, public = create_private_public(filename, -1)

    public_data = public[:, 0]
    public_label = public[:, 1]
    
    sim = []
    for ind, i in enumerate(private):
        start = time.time()
        f1 = np.array(similarity([i[0]], public_data)['f1'])
        index = np.argsort(f1)[::-1][:5]
        sim.append([list(index), list(f1[index])])
        end = time.time()
        print(f'{ind} took {round(end-start, 2)} seconds.\n')

    dic = {'private': private, 'public': public, 'similar': sim}
    file_name = "/Users/sarinaxi/Desktop/Thesis/StudentTeacher/sim_10.pkl"

    open_file = open(file_name, "wb")
    pickle.dump(dic, open_file)
    open_file.close()