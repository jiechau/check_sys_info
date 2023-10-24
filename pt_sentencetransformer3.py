size=100
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('***torch.cuda.is_available()', torch.cuda.is_available())
print('***device', device)

from torchvision import datasets
train_data = datasets.FashionMNIST(
    root='/tmp/data', 
    train=True,
    download=True
)

# Get a random sample of 10 images 
import numpy as np    
rnd_inds = np.random.randint(0, len(train_data), size)
# images = [np.array(train_data[i][0]) for i in rnd_inds] # 
images = [train_data[i][0] for i in rnd_inds] # already <class 'PIL.Image.Image'>
labels = [train_data[i][1] for i in rnd_inds]

'''
# Display the images 
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 10, figsize=(20,2))
for i, ax in enumerate(axes):
    ax.imshow(images[i], cmap='gray')
    ax.set_title(labels[i])
    ax.axis('off')
plt.show()
'''

# https://www.sbert.net/examples/applications/paraphrase-mining/README.html

from sentence_transformers import SentenceTransformer, util
#Load CLIP model
model = SentenceTransformer('clip-ViT-B-32', device=device)

import time
start = time.time()


similarity = util.paraphrase_mining(model, images)



end = time.time()


print(similarity)
print(labels)
print('***torch.cuda.is_available()', torch.cuda.is_available())
print('***device', device)
print('time: ', end - start)






