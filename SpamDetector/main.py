from spam_detector import SpamDetector
from model import PT_Arch
from transformers import BertModel, OpenAIGPTModel, BertTokenizerFast, AutoTokenizer
import torch
import yaml
import os 
from plotting_analytics import plot_loss_acc, model_performance

import warnings
warnings.filterwarnings('ignore')

#spacy.load('en_core_web_sm')
if __name__ == '__main__':
    # check using GPU
    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())

    # define variables 
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    args = config['spamDetector']
    architecture = args['architecture']
    path = f'{architecture}_saved_weights.pt'
    type_ = 'sms' if args['sms'] else 'email'
    diff = 'easy' if args['easy'] else 'hard'
    folder = f'test_custom_weight_with_softmax_celoss2_{type_}_{architecture}_{diff}'
    device = torch.device("mps")

    if not os.path.exists(folder):
        os.makedirs(folder)
    if architecture == 'gpt':
        arch = OpenAIGPTModel.from_pretrained('openai-gpt')
        tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
    if architecture == 'bert':
        arch = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = PT_Arch(arch, args['dropout'], architecture).to(device)
    # freeze all the parameters
    for param in arch.parameters():
        param.requires_grad = False
    
    # train model with training and validation datasets
    spamDetector = SpamDetector(model=model, tokenizer=tokenizer, device=device, lr=args['lr'], 
                                batch_size=args['batch_size'], splits=args['splits'], epochs=args['epochs'], 
                                data_filename=args['data_file'], index=args['index'], weight_path=path, folder=folder,
                                sms=args['sms'], easy=args['easy'])
    train_losses, train_acc, valid_losses, valid_accs = spamDetector.run()
    spamDetector.model.load_state_dict(torch.load(f'{folder}/{path}'))
    
    # plot curves and evaluate model on test set 
    plot_loss_acc(train_losses, valid_losses, 'Loss', folder)
    plot_loss_acc(train_acc, valid_accs, 'Acc', folder)
    model_performance(args, spamDetector.model, spamDetector.test_data[0], spamDetector.test_data[1], spamDetector.test_data[2], device, folder)

