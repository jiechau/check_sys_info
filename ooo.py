from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

#Our sentences we like to encode
s = 'This framework generates embeddings for each input sentence'
sentences = [s for i in range(10_000)]
print(len(sentences))

import torch
import time
start = time.time()
#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)
#
end = time.time()
print(end - start)
print('cuda: ' + str(torch.cuda.is_available()))


#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    #print("Sentence:", sentence)
    #print("Embedding:", embedding)
    #print("")
    pass
