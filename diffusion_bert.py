import math
from torch.utils.data.dataset import Dataset
import csv
from transformers import AutoModelForPreTraining,AutoModelForMaskedLM
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F 
import random
class ROCstory(Dataset): 
    def __init__(self,csv_dir,init_model,max_len):
        self.tokenizer = AutoTokenizer.from_pretrained(init_model)
        with open(csv_dir,'r') as f:
            story_teller = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
            self.story = list(story_teller)
        self.max_len = max_len
        self.tokenizer.model_max_length = max_len
    def __getitem__(self, index):
        #index = 0
        #index = random.randint(0,9)
        story = "".join(self.story[index][2:])
        from_tokenizer = self.tokenizer(story,padding="max_length",truncation = True,return_tensors="pt")
        input_ids = from_tokenizer["input_ids"].squeeze_().long()
        token_type_ids = from_tokenizer["token_type_ids"].squeeze_().long()
        attention_mask = from_tokenizer["attention_mask"].squeeze_().long()
        return input_ids,token_type_ids,attention_mask
    def __len__(self):
        return len(self.story)

class e2e(Dataset): 
    def __init__(self,csv_dir,init_model,max_len):
        self.tokenizer = AutoTokenizer.from_pretrained(init_model)
        with open(csv_dir,'r') as f:
            story_teller = f.readlines()
            self.story = list(story_teller)
        self.max_len = max_len
        self.tokenizer.model_max_length = max_len
    def __getitem__(self, index):
        #index = 0
        #index = random.randint(0,9)
        story = self.story[index].split("||")[-1].strip()
        from_tokenizer = self.tokenizer(story,padding="max_length",truncation = True,return_tensors="pt")
        input_ids = from_tokenizer["input_ids"].squeeze_().long()
        token_type_ids = from_tokenizer["token_type_ids"].squeeze_().long()
        attention_mask = from_tokenizer["attention_mask"].squeeze_().long()
        return input_ids,token_type_ids,attention_mask
    def __len__(self):
        return len(self.story)

class diffusion_bert(nn.Module):
    def __init__(self,init_model,max_len,max_step) -> None:
        super().__init__()
        if "bert-base" in init_model:
            self.model = AutoModelForMaskedLM.from_pretrained(init_model)
            freezed_w = [self.model.bert.embeddings.token_type_embeddings.weight,self.model.bert.embeddings.word_embeddings.weight] #self.model.bert.embeddings.LayerNorm.weight, self.model.bert.embeddings.LayerNorm.bias
        else:
            self.model = AutoModelForPreTraining.from_pretrained(init_model)
            freezed_w = [self.model.cls.seq_relationship.bias, self.model.cls.seq_relationship.weight, self.model.bert.pooler.dense.bias, self.model.bert.pooler.dense.weight, self.model.bert.embeddings.token_type_embeddings.weight,self.model.bert.embeddings.word_embeddings.weight] #self.model.bert.embeddings.LayerNorm.weight, self.model.bert.embeddings.LayerNorm.bias
        self.max_len = max_len
        self.max_step = max_step
        self.time_embed = nn.Embedding(max_step,self.model.config.hidden_size)
        #self.layernorm = nn.LayerNorm(self.model.config.hidden_size, eps=self.model.config.layer_norm_eps)
        for p in  freezed_w:
            p.requires_grad = False
        nn.init.constant_(self.time_embed.weight, 0)
    def forward(self,input_ids,token_type_ids,attention_mask,t =None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        
        position_ids = self.model.bert.embeddings.position_ids[:, 0 : seq_length]
        position_embeddings = self.model.bert.embeddings.position_embeddings(position_ids)
       
        with torch.no_grad():
            word_emb = self.model.bert.embeddings.word_embeddings(input_ids)
            #print(word_emb.shape)
            token_type_embeddings = self.model.bert.embeddings.token_type_embeddings(token_type_ids)
            if t is None:
                diffusion_steps = torch.randint(0,self.max_step,size = (input_shape[0],),device=input_ids.device)
            else:
                diffusion_steps = torch.ones(size = (input_shape[0],),device=input_ids.device).long()*t

            noise = torch.randn_like(word_emb)/math.sqrt(self.model.config.hidden_size)
            alpha = 1 - torch.sqrt((diffusion_steps+1)/self.max_step).view(-1,1,1)
            noisy_word = torch.sqrt(alpha)*word_emb+torch.sqrt(1-alpha)*noise + token_type_embeddings
            
        time_embedding = self.time_embed(diffusion_steps).unsqueeze(1)
        noisy_word = noisy_word+position_embeddings+time_embedding
        
        #noisy_word = self.layernorm(noisy_word)
        noisy_word = self.model.bert.embeddings.LayerNorm(noisy_word)

        extended_attention_mask = self.model.bert.get_extended_attention_mask(attention_mask, input_shape)
        
        encoder_outputs = self.model.bert.encoder(
            noisy_word,
            attention_mask=extended_attention_mask,
            head_mask=[None] * self.model.config.num_hidden_layers
        )
        sequence_output = encoder_outputs[0]
        prediction_scores = self.model.cls.predictions(sequence_output)
        loss = F.cross_entropy(prediction_scores.view(-1, self.model.config.vocab_size),input_ids.flatten(),ignore_index=0)
        
        #loss = F.smooth_l1_loss(sequence_output,word_emb)
        return loss,prediction_scores,diffusion_steps

    def test_pretrained(self,input_ids,token_type_ids,attention_mask):
        loss,prediction_scores,diffusion_steps = self.forward(input_ids,token_type_ids,attention_mask,0)
        return loss,prediction_scores,diffusion_steps


    @torch.no_grad()
    def sampler(self,device,k=10,N=128):
        import time
        
        start_time = time.time()
        # mean,std = stats
        # mean = torch.tensor(mean).view(1,3,1,1)
        # std = torch.tensor(std).view(1,3,1,1)    
        noisy_word = torch.normal(0,1,(N,self.max_len,self.model.config.hidden_size)).to(device) / math.sqrt(self.model.config.hidden_size)
        token_type_ids = torch.zeros(N,self.max_len).long().to(device)
        attention_mask = torch.ones(N,self.max_len).long().to(device)
        extended_attention_mask = self.model.bert.get_extended_attention_mask(attention_mask, attention_mask.shape)

        position_ids = self.model.bert.embeddings.position_ids[:, 0 : self.max_len]
        position_embeddings = self.model.bert.embeddings.position_embeddings(position_ids)
        token_type_embeddings = self.model.bert.embeddings.token_type_embeddings(token_type_ids)
        for t in range(self.max_step-1,0,-k):
        #for t in range(1999,0,-1):

            #prepare time emb
            diffusion_steps = torch.ones(size = (N,),device=device).long()*t
            time_embedding = self.time_embed(diffusion_steps).unsqueeze(1)

            model_input = noisy_word+position_embeddings+token_type_embeddings+time_embedding
            model_input = self.model.bert.embeddings.LayerNorm(model_input)
            #denoise
            encoder_outputs = self.model.bert.encoder(
                model_input,
                attention_mask=extended_attention_mask,
                head_mask=[None] * self.model.config.num_hidden_layers
            )
            sequence_output = encoder_outputs[0]
            prediction_scores = self.model.cls.predictions(sequence_output)

            #clamp
            # pred = torch.argmax(prediction_scores,-1).long()
            # denoised_word = self.model.bert.embeddings.word_embeddings(pred)
            denoised_word = prediction_scores.softmax(-1) @ self.model.bert.embeddings.word_embeddings.weight.unsqueeze(0)
        
            #DDIM
            alpha_tk = 1 - math.sqrt((t+1-k)/self.max_step)#+1e-5
            alpha_t = 1 - math.sqrt((t+1)/self.max_step)+1e-5
            noise = (noisy_word - math.sqrt(alpha_t)*denoised_word)/math.sqrt(1-alpha_t)
            noisy_word = math.sqrt(alpha_tk)*(noisy_word/math.sqrt(alpha_t) + (math.sqrt((1-alpha_tk)/alpha_tk) - math.sqrt((1-alpha_t)/alpha_t))*noise)
            #noisy_word = math.sqrt(alpha_tk)*denoised_word + math.sqrt(1-alpha_tk)*noise
            print(f"\rnoise level {t}  {time.time()-start_time:.2f}",end='')
        
        pred = torch.argmax(prediction_scores,-1).long()
        return pred

if __name__ == "__main__":
    initializing = '/home/haroldjia/Harold_Laptop_WSL/Temp/diffusion_model/diffusion_bert/bert-base-uncased'
    max_len = 64
    diff_step = 2000
    device = torch.device('cuda')
    model = diffusion_bert(initializing,max_len,diff_step)
    state = torch.load("/home/haroldjia/Harold_Laptop_WSL/Temp/diffusion_model/diffusion_bert/Saved_Models/20220822bert_diffusion/bestloss.pkl")
    model.load_state_dict(state,strict=True)
    model = model.to(device)
    model.eval()
    
    test_set = ROCstory("/home/haroldjia/Harold_Laptop_WSL/Temp/diffusion_model/diffusion_bert/ROCstory_test.csv",init_model=initializing,max_len=max_len)
    out = model.sampler(device,1,32)
    for s in out:
        sample = test_set.tokenizer.decode(s.cpu().flatten())
        print(sample)        

    #debug: test training
    test_set = ROCstory("/home/haroldjia/Harold_Laptop_WSL/Temp/diffusion_model/diffusion_bert/ROCstory_test.csv",init_model=initializing,max_len=max_len)
    input_ids,token_type_ids,attention_mask = test_set.__getitem__(3)
    input_ids = input_ids.unsqueeze_(0).to(device)
    token_type_ids = token_type_ids.unsqueeze_(0).to(device)
    attention_mask = attention_mask.unsqueeze_(0).to(device)
    for t in range(0,2000,100):
        with torch.no_grad():
            loss,prediction_scores,diffusion_steps = model(input_ids,token_type_ids,attention_mask,t)
        pred = torch.argmax(prediction_scores,-1).cpu()
        pred = test_set.tokenizer.decode(pred.flatten())
        print(pred)
