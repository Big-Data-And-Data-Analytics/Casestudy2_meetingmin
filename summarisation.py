#https://medium.com/rocket-mortgage-technology-blog/conversational-summarization-with-natural-language-processing-c073a6bcaa3a
import os
import json

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from scipy import stats

import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForMaskedLM, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl

from torch.utils.data.dataset import TensorDataset, random_split

val_path = 'C:/Users/chait/Documents/minutesmeet/Dialogue_summary_dataset/val.json'
test_path = 'C:/Users/chait/Documents/minutesmeet/Dialogue_summary_dataset/test.json'
train_path = 'C:/Users/chait/Documents/minutesmeet/Dialogue_summary_dataset/train.json'
with open(val_path, encoding="utf8") as in_file:
    val = json.load(in_file)
    in_file.close()
    
with open(test_path, encoding="utf8") as in_file:
    test = json.load(in_file)
    in_file.close()
    
with open(train_path, encoding="utf8") as in_file:
    train = json.load(in_file)
    in_file.close()
    
data = train + test + val

df = pd.DataFrame(data)
df['dialogue'] = df['dialogue'].str.replace('\r', '')
df['dialogue'] = df['dialogue'].str.replace('\n', ' ')
df['summary'] = df['summary'].str.replace('\r', '')
df['summary'] = df['summary'].str.replace('\n', ' ')

from transformers import T5Tokenizer

t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_df = df.sample(1000)

dialogue = t5_df['dialogue'].values.tolist()
summary = t5_df['summary'].values.tolist()

data = t5_tokenizer.prepare_seq2seq_batch(
    src_texts=[f'summarize: {d}' for d in dialogue], 
    tgt_texts=summary, 
    padding='max_length', 
    truncation=True, 
    return_tensors='pt'
)

from torch.utils.data import TensorDataset
tdata = TensorDataset(
    data['input_ids'], 
    data['attention_mask'], 
    data['labels']
)

import torch
from torch.utils.data import random_split

train_size = int(data['input_ids'].shape[0] * 0.80)
test_size = int(data['input_ids'].shape[0] * 0.10)
val_size = int(data['input_ids'].shape[0]) - train_size - test_size

train, test, val = random_split(dataset=tdata, lengths=(train_size, test_size, val_size))

torch.save(train, 'C:/Users/chait/Documents/minutesmeet/processed/t5_train_dataset.pt')
torch.save(test, 'C:/Users/chait/Documents/minutesmeet/processed/t5_test_dataset.pt')
torch.save(val, 'C:/Users/chait/Documents/minutesmeet/processed/t5_val_dataset.pt')

class T5LightningModule(pl.LightningModule):
    def __init__(
        self,
        pretrained_nlp_model: str,
        train_dataset: str,
        test_dataset: str,
        val_dataset: str,
        batch_size: int,
        learning_rate: float = 3e-05,
    ):
        """
        A Pytorch-Lightning Module that trains Bart from the  HuggingFace transformers
        library.

        :param pretrained_nlp_model: (str) the name of the pretrained mode you want to use.
        :param train_dataset: (str) path to pytorch dataset containing train data.
        :param test_dataset: (str) path to pytorch dataset containing test data.
        :param val_dataset: (str) path to pytorch dataset containing validation data.
        :param batch_size: (int) Number of data points to pass per batch in the train, test, and validation sets.
        :param learning_rate: (float) Initial Learning Rate to set.
        :returns: None
        """
        super().__init__()

        self.batch_size = int(batch_size)
        self.train_dataset = str(train_dataset)
        self.test_dataset = str(test_dataset)
        self.val_dataset = str(val_dataset)
        self.hparams.learning_rate = learning_rate
        
        self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_nlp_model)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_nlp_model)
    def forward(self, x):
        
        # Run through NLP Model
        output = self.t5(**x)
        return output

    def training_step(self, batch, batch_idx):

        input_ids, attn_mask, labels = batch

        x = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "return_dict": True,
        }

        # Run through NLP Model
        out = self.t5(**x)

        loss = out["loss"]
        print(f"current_epoch: {self.current_epoch};")
        print(f"global_step: {self.global_step};")
        print(f"train_loss: {loss};")
        print(f"learning_rate: {self.hparams.learning_rate};")

        
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss
    def validation_step(self, batch, batch_idx):
        input_ids, attn_mask, labels = batch

        x = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "return_dict": True,
        }

        # Run through NLP Model
        out = self.t5(**x)
        loss = out["loss"]
        
        
        print(f"val_loss: {loss};")
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)

        # Generating Example Summary
        if batch_idx == len(self.val_dataloader())-1:
            
            outputs = self.t5.generate(input_ids)
            predictions = self.tokenizer.batch_decode(
                outputs, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            
            references = self.tokenizer.batch_decode(
                labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for idx, pred in enumerate(predictions):
                
                self.logger.experiment.add_text(
                    tag='example_summaries',
                    text_string=f'''
                    Model Summary: {predictions[idx]}
                
                    Target Summary: {references[idx]}''',
                    global_step=self.global_step,
                )
        return loss            
    def test_step(self, batch, batch_idx):
        input_ids, attn_mask, labels = batch

        x = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "return_dict": True,
        }

        # Run through NLP Model
        out = self.t5(**x)

        loss = out["loss"]
        print(f"test_loss: {loss};")

        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True) 

        return loss

    def configure_optimizers(self):
        """
        Recreating the same Adam optimizer used in the author's code.
        """

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
        
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=20000)
        print(f'scheduler: {scheduler}')
        gen_sched = {'scheduler': scheduler, 'interval': 'step'}
        
        return [optimizer], [gen_sched]
    def train_dataloader(self):
        return DataLoader(
            torch.load(self.train_dataset), shuffle=True, batch_size=self.batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            torch.load(self.val_dataset), shuffle=False, batch_size=self.batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            torch.load(self.test_dataset), shuffle=True, batch_size=self.batch_size
        )

model = T5LightningModule(
    train_dataset='C:/Users/chait/Documents/minutesmeet/processed/t5_train_dataset.pt',
    test_dataset='C:/Users/chait/Documents/minutesmeet/processed/t5_test_dataset.pt',
    val_dataset='C:/Users/chait/Documents/minutesmeet/processed/t5_val_dataset.pt',
    pretrained_nlp_model='t5-small',
    batch_size=4,
    learning_rate=3e-05,
)

trainer_params = {
    "max_epochs": 0,
    "default_root_dir": 'C:/Users/chait/Documents/minutesmeet/models/t5',
    "gpus": 0,
    #"logger": tb_logger,
    #"early_stop_callback": early_stop,
    #"checkpoint_callback": model_checkpoint,
    #"callbacks": [lr_logger],
    #"precision": 16,
    'fast_dev_run': False
}

print(f"Trainer Params: {trainer_params}")

trainer = pl.Trainer(**trainer_params)
trainer.fit(model)



#model.load_state_dict(torch.load('C:/Users/chait/Documents/casestudy2/models/t5/lightning_logs/version_0/checkpoints/epoch=0-step=199.ckpt', map_location=torch.device('cpu'))['state_dict'])

tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')

dialogue = df['dialogue'].values.tolist()
summary = df['summary'].values.tolist() 

dialogue_tokens = t5_tokenizer.prepare_seq2seq_batch(dialogue, padding='longest', truncation=False, return_tensors='np')

summary_tokens = t5_tokenizer.prepare_seq2seq_batch(summary, padding='longest', truncation=False, return_tensors='np')

num_dialogues, longest_dialogue = dialogue_tokens['input_ids'].shape
num_summaries, longest_summary = summary_tokens['input_ids'].shape
assert num_dialogues == num_summaries

#print(f'The longest dialogue is: {longest_dialogue}\n')
#print(f'The longest summary is: {longest_summary}')

model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')

text_file = open("output/summary.txt", "r")
lines = text_file.readlines()
#print(lines)

def summ_pred():
    dialogue = lines
    "Chaithra: Hello. Sayli: Hello. Chaithra: We are having this call for information system module. Sayli: Lets divide the work so we can get going. Chaithra:I will do web scraping. Sayli: I will do data modelling. Chaithra: Bye. Sayli: Bye. "
    print(dialogue)
    batch = tokenizer.prepare_seq2seq_batch(dialogue, truncation=True, padding='longest', return_tensors='pt')
    translated = model.generate(**batch) 

    #print(f'dialogue:\n{dialogue}\n')
    print(f'model summary:\n{tokenizer.batch_decode(translated, skip_special_tokens=True)[0]}') 
    return(f'model summary:\n{tokenizer.batch_decode(translated, skip_special_tokens=True)[0]}') 

if __name__=='__main__':
    summ_pred()

