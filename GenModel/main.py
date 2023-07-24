import sys
sys.path.insert(0, '/Users/sarinaxi/Desktop/Thesis')

import os
from GenModel.process_data import process_data
from GenModel.generative_model import GenerativeModel
import torch
import numpy as np
from transformers import AutoTokenizer, GPT2LMHeadModel

if __name__ == '__main__':
    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())

    device = torch.device("mps")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    train_dataloader, valid_dataloader, test_dataloader = process_data(tokenizer, 0.3, 32, 
                                                                       'SpamDetector/data/smsSpam/SMSSpamCollection.txt', 
                                                                       -2)
    path = 'saved_weights.pt'
    folder = 'test'
    if not os.path.exists(folder):
        os.makedirs(folder)
    model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
    model = model.to(device)
    generativeModel = GenerativeModel(model=model, tokenizer=tokenizer, device=device, lr=0.0007, 
                                batch_size=32, splits=0.3, epochs=20, 
                                data_filename='SpamDetector/data/smsSpam/SMSSpamCollection.txt', 
                                index=-2, weight_path=path, folder=folder)
    train_losses, valid_losses = generativeModel.run()
    generativeModel.model.load_state_dict(torch.load(f'{folder}/{path}'))

    # SAMPLING
    spam = tokenizer.encode('spam ', return_tensors='pt')
    ham = tokenizer.encode('ham ', return_tensors='pt')
    np.random.seed(0)

    model.cpu()
    spam_outputs = model.generate(
        ham,
        do_sample=True, 
        max_length=50, 
        top_k=50, 
        top_p=0.95, 
        num_return_sequences=5, 
    )

    print("Output:\n" + 100 * '-')
    for i, sample_output in enumerate(spam_outputs):
        print("{}: {}".format(i, tokenizer.decode(sample_output)))
