
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('***torch.cuda.is_available()', torch.cuda.is_available())
print('***device', device)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
#model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:0')

#Our sentences we like to encode
s = 'This framework generates embeddings for each input sentence'
sentences = [s for i in range(10_000)]
print(len(sentences))

import time
start = time.time()
#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)
#
end = time.time()
print(end - start)
print(device, str(torch.cuda.is_available()))


#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    #print("Sentence:", sentence)
    #print("Embedding:", embedding)
    #print("")
    pass
