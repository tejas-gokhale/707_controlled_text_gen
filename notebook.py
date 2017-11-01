import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import numpy as np
from konlpy.tag import Mecab;tagger=Mecab()
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')


# * https://arxiv.org/pdf/1703.00955.pdf

USE_CUDA = torch.cuda.is_available()
print("Using CUDA: ", USE_CUDA)

data = open('data/ratings_train.txt','r',encoding='utf-8').readlines()
data = data[1:]
data = [[d.split('\t')[1],d.split('\t')[2][:-1]] for d in data]


distibution = [d[1] for d in data]


positive = [d for d in data if d[1]=="1"]
negative = [d for d in data if d[1] =="0"]


data = random.sample(positive,1000) + random.sample(negative,1000)


SEQ_LENGTH=15

train=[]

for t in data:
    t0 = t[0]
    t0 = t0.replace("<br>","")
    t0 = t0.replace("/","")
    
    token0 = tagger.morphs(t0)
    
    if len(token0)>=SEQ_LENGTH:
        token0= token0[:SEQ_LENGTH-1]
    token0.append("<EOS>")

    while len(token0)<SEQ_LENGTH:
        token0.append('<PAD>')
    
    train.append([token0,token0,t[1]])


word2index={"<PAD>":0,"<SOS>":1,"<EOS>":2,"<UNK>":3}

for t in train:
    for token in t[0]:
        if token not in word2index:
            word2index[token]=len(word2index)

index2word = {v:k for k,v in word2index.items()}


def prepare_sequence(seq, to_ix):
    idxs = list(map(lambda w: to_ix[w] if w in to_ix.keys() else to_ix["<UNK>"], seq))
    tensor = Variable(torch.LongTensor(idxs)).cuda() if USE_CUDA else Variable(torch.LongTensor(idxs))
    return tensor


flatten = lambda l: [item for sublist in l for item in sublist]


train_x=[]
train_y=[]
code_labels=[]
lengths=[]
for tr in train:
    temp = prepare_sequence(tr[0], word2index)
    temp = temp.view(1,-1)
    train_x.append(temp)

    temp2 = prepare_sequence(tr[1],word2index)
    temp2 = temp2.view(1,-1)
    train_y.append(temp2)
    
    length = [t for t in tr[1] if t !='<PAD>']
    lengths.append(len(length))
    code_labels.append(Variable(torch.LongTensor([int(tr[2])])).cuda() if USE_CUDA else Variable(torch.LongTensor([int(tr[2])])))



train_data = list(zip(train_x,train_y,code_labels))


def getBatch(batch_size,train_data):
    random.shuffle(train_data)
    sindex=0
    eindex=batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        x,y,c = zip(*batch)
        x,y,c = torch.cat(x),torch.cat(y),torch.cat(c)
        temp = eindex
        eindex = eindex+batch_size
        sindex = temp
        
        yield (x,y,c)




class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size,latent_size=10,n_layers=1):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.Wmu= nn.Linear(hidden_size,latent_size)
        self.Wsigma = nn.Linear(hidden_size,latent_size)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,batch_first=True)
    
    def reparametrize(self, mu, log_var):
        """"z = mean + eps * sigma where eps is sampled from N(0, 1)."""
        eps = Variable(torch.randn(mu.size(0), mu.size(1))).cuda() if USE_CUDA else Variable(torch.randn(mu.size(0), mu.size(1)))
        z = mu + eps * torch.exp(log_var/2)    # 2 for convert var to std
        return z
    
    def forward(self, input,train=True):
        hidden = Variable(torch.zeros(self.n_layers, input.size(0), self.hidden_size)).cuda() if USE_CUDA else Variable(torch.zeros(self.n_layers, input.size(0), self.hidden_size))
        
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        mu = self.Wmu(hidden[-1])
        log_var = self.Wsigma(hidden[-1])
        z = self.reparametrize(mu, log_var)
        
        return z,mu,log_var




class Generator(nn.Module):
    def __init__(self, hidden_size, output_size,latent_size=10,code_size=2, n_layers=1):
        super(Generator, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        #self.Wz = nn.Linear(latent_size+code_size,hidden_size)
        self.Wz = nn.Linear(latent_size,hidden_size)
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        #self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers,batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, input,latent,code,lengths,seq_length,training=True):
        

        embedded = self.embedding(input)
        #embedded = self.dropout(embedded)
       
        # h0
        #latent_code = torch.cat((latent,code),1) # z,c
        #hidden = self.tanh(self.Wz(latent_code)).view(self.n_layers,input.size(0),-1) 
        hidden = self.tanh(self.Wz(latent)).view(self.n_layers,input.size(0),-1) 
        decode=[]
        # Apply GRU to the output so far
        for i in range(seq_length):
            
            _, hidden = self.gru(embedded, hidden)
            score = self.out(hidden.view(hidden.size(0)*hidden.size(1),-1))
            softmaxed = F.log_softmax(score)
            decode.append(softmaxed)
            _,input = torch.max(softmaxed,1)
            embedded = self.embedding(input.unsqueeze(1))
            #embedded = self.dropout(embedded)
        
        # 요고 주의! time-step을 column-wise concat한 후, reshape!!
        scores = torch.cat(decode,1)
        
        return scores.view(input.size(0)*seq_length,-1)




class  Discriminator(nn.Module):
    
    def __init__(self, embed_num,embed_dim,class_num,kernel_num,kernel_sizes,dropout):
        super(Discriminator,self).__init__()
        #self.args = args
        
        V = embed_num # num of vocab
        D = embed_dim # dimenstion of word vector
        C = class_num # num of class
        Ci = 1
        Co = kernel_num # 100
        Ks = kernel_sizes # [3,4,5]

        self.embed = nn.Embedding(V, D)
        #self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        
        # kernal_size = (K,D) : D는 단어 벡터 길이라 픽스, K 사이즈만큼 슬라이딩, 스트라이드는 1
        
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x


    def forward(self, x,train=True):
        x = self.embed(x) # (N,W,D)
        
        #if self.args.static:
        #    x = Variable(x)

        x = x.unsqueeze(1) # (N,Ci,W,D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)


        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        if train:
            x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc1(x) # (N,C)
        return logit




HIDDEN_SIZE = 300
LATENT_SIZE = 10
CODE_SIZE = 2
BATCH_SIZE=32
STEP=500
KTA = 0.0
LEARNING_RATE=0.001




encoder =  Encoder(len(word2index), HIDDEN_SIZE,LATENT_SIZE, 2)
generator = Generator(HIDDEN_SIZE,len(word2index),LATENT_SIZE,CODE_SIZE)
discriminator = Discriminator(len(word2index),100,2,30,[3,4,5],0.8)
if USE_CUDA:
    encoder = encoder.cuda()
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    
Recon = nn.CrossEntropyLoss(ignore_index=0)


enc_optim= torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
gen_optim = torch.optim.Adam(generator.parameters(),lr=LEARNING_RATE)
dis_optiom = torch.optim.Adam(discriminator.parameters(),lr=LEARNING_RATE)


# ## 1. Initialize base VAE 



for step in range(STEP):
    for i,(x,y,c) in enumerate(getBatch(BATCH_SIZE,train_data)):

        encoder.zero_grad()
        generator.zero_grad()

        generator_input = Variable(torch.LongTensor([[word2index['<SOS>']]*BATCH_SIZE])).transpose(1,0)

        if USE_CUDA:
            generator_input = generator_input.cuda()

        latent, mu, log_var = encoder(x)
        
        # 이 때, 코드는 prior p(c)에서 샘플링한다 되있는데, 이게 맞나.. 일단 유니폼 가정
        code = Variable(torch.randn([BATCH_SIZE,2]).uniform_(0,1)).cuda() if USE_CUDA else Variable(torch.randn([BATCH_SIZE,2]).uniform_(0,1))

        score = generator(generator_input,latent,code,lengths,SEQ_LENGTH)
        recon_loss=Recon(score,y.view(-1))
        kld_loss = torch.sum(0.5 * (mu**2 + torch.exp(log_var) - log_var -1))

    #     KL_COST_ANNEALING
        cost_annealing_check = recon_loss.data.cpu().numpy()[0] if USE_CUDA else recon_loss.data.numpy()[0]
        if cost_annealing_check<1.5:
            KTA = 0.5 # KL cost term annealing
        elif cost_annealing_check<1.0:
            KTA = 0.75
        elif cost_annealing_check<0.5:
            KTA = 1.0
        else:
            KTA = 0.0
            
        ELBO = recon_loss+KTA*kld_loss

        ELBO.backward()

        torch.nn.utils.clip_grad_norm(encoder.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm(generator.parameters(), 5.0)

        gen_optim.step()
        enc_optim.step()
    
    # KL term Anealing
    #KTA+=1/STEP
    #KTA = round(KTA,3)
    
    if step % 10==0:
        elbo_for_print = ELBO.data.cpu().numpy()[0] if USE_CUDA else ELBO.data.numpy()[0]
        recon_for_print = recon_loss.data.cpu().numpy()[0] if USE_CUDA else recon_loss.data.numpy()[0]
        kld_for_print = kld_loss.data.cpu().numpy()[0] if USE_CUDA else kld_loss.data.numpy()[0]
        print("[%d/%d] ELBO : %.4f , RECON : %.4f & KLD : %.4f" % (step,STEP,elbo_for_print,
                                                                              recon_for_print,
                                                                              kld_for_print))




torch.save(generator.state_dict(),'models/generator.pkl')
torch.save(encoder.state_dict(),'models/encoder.pkl')




generator_input = Variable(torch.LongTensor([[word2index['<SOS>']]*1])).transpose(1,0)
if USE_CUDA:
    generator_input = generator_input.cuda()

latent = Variable(torch.randn([1,10])).cuda() if USE_CUDA else Variable(torch.randn([1,10]))
code = Variable(torch.randn([1,2]).uniform_(0,1)).cuda() if USE_CUDA else Variable(torch.randn([1,2]).uniform_(0,1))
recon = generator(generator_input,latent,code,15,SEQ_LENGTH,False)

v,i = torch.max(recon,1)

decoded=[]
for t in range(i.size()[0]):
    decoded.append(index2word[i.data.cpu().numpy()[t] if USE_CUDA else i.data.cpu().numpy()[t]])

print('A: ', ' '.join([i for i in decoded if i !='<PAD>' and i != '<EOS>'])+'\n')


# # TODO 

# * 우선 VRAE 초기화가 잘 되는지 체크(kl cost annealing 제대로)
# * Encoder 진짜 length만
# * 다른 로스들도 실험
# * wakeup-sleep 적용
