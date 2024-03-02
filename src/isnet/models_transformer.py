from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel
from transformers import LlamaModel, LlamaConfig
# from isnet.configs import GPTConfig
@dataclass
class GPTConfig:
    block_size: int = 1#00
    vocab_size: int = 2 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    # n_embd: int = 128
    dropout: float = 0.
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    out_d: int=1


# Initializing a model (with random weights) from the configuration
class Zero_module(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 0.

class LinearEmbedding(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.l = nn.Linear(dim_in, dim_out)
    def forward(self,x):
        x = self.l(x)
        return x
class Model_GPT(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.gpt = GPT2Model(configs)
        self.gpt.wte = LinearEmbedding(configs.vocab_size, configs.n_embd)
        self.gpt.wpe = Zero_module()
        self.lm_head = nn.Linear(configs.n_embd, 1)
    def forward(self, x):
        x = x.view(x.shape[0],1,x.shape[-1])
        x_embd = self.gpt.wte(x)
        inputs = {'input_ids': None, 'attention_mask': torch.ones_like(x[:,:,:-1])}
        x = self.gpt(**inputs,inputs_embeds = x_embd)[0] 
        return self.lm_head(x).squeeze(1)
    def configure_optimizers(self, train_config):#, weight_decay, learning_rate, betas, device_type):

        optimizer = torch.optim.RMSprop(self.parameters(), lr=train_config.lr)#, betas=betas, **extra_args)
        # optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        # print(f"using fused AdamW: {use_fused}")

        return optimizer
    
configs = GPT2Config(vocab_size=2, n_embd=768)
x = torch.randn((4,2))
model = Model_GPT(configs)

print(model)
print(model(x).shape)
# # model = GPT2Model(configuration)


# model.wte = LinearEmbedding(2,768)
# model.wpe = Zero_module()
# print(model)
# x =dict({'input_ids': torch.randn((1,1,2)), 'attention_mask': torch.ones((1,1))})
# # x['input_ids'] = torch.randn((2,1,2))
# # x['attention_mask'] = torch.ones((2,1))
# idx = x['input_ids']
# pos =idx
# tok_emb = model.wte(idx) # token embeddings of shape (b, t, n_embd)
# pos_emb = model.wpe(pos) # position embeddings of shape (t, n_embd)

# inputs_embeds = model.wte(idx)
# x['input_ids'] = None
# print(inputs_embeds.shape)
# print(model(**x,inputs_embeds = inputs_embeds)[0])

# from transformers import GPT2Tokenizer, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# print(inputs)
# print(model)
# # print(model.wte(inputs['input_ids']).shape)
# with torch.no_grad():
#     logits = model(**inputs)
# print(logits[0].shape)
# import torch.nn.functional as F
# probs = F.softmax(logits, dim=-1)
# idx_next = torch.multinomial(logits, num_samples=1)
# print(model)
# print(tokenizer.decode(idx_next))