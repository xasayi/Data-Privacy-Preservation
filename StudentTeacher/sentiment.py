import pandas as pd
import numpy as np
import torch
import sys
sys.path.insert(0, '/Users/sarinaxi/Desktop/Thesis')
from StudentTeacher.process_data import process_data_bert
from StudentTeacher.main import args_and_init, pre_train, check_ptmodel
from StudentTeacher.model import EmbedModel, LSTMModelMulti
device = torch.device("cpu")

if __name__ == '__main__':
    # define variables
    
    filename, st_args, te_args, args = args_and_init(True, True, "StudentTeacher/sentiment_config.yaml")
    df = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/StudentTeacher/data/sentiment_data/sentiment_data2.csv')
    df = df.drop_duplicates(subset=['data']).reset_index(drop=True)
    map = {df['type'].unique()[i]:i for i in range(len(df['type'].unique()))}
    print(map)

    train_loader, valid_loader, test_data, weight = process_data_bert(df=df, splits=args['splits'], 
                                                                 bs=args['batch_size'], max_seq=args['input_size'], 
                                                                 downsample=args['downsample'], map=map)
    print('here')
    #student = LSTMModel(30522, st_args['embed_size'], st_args['hidden_size'], st_args['dropout']).to(device)
    teacher = LSTMModelMulti(6, 30522, te_args['embed_size'], te_args['hidden_size'], te_args['dropout']).to(device)
    pre_train(teacher, te_args, df, map)
    check_ptmodel(teacher, te_args, df, True, map)
    

