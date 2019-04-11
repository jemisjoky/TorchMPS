#!/usr/bin/env python
# coding: utf-8
import Hasyv2Dataset 
import matplotlib.pyplot as plt
from torchvision import transforms

def showImage(path):
    """Show an image given its corresponding path"""
    plt.figure()
    plt.imshow(path)
    plt.show()
    

#Plot some images
hasy_dataset = Hasyv2Dataset.Hasyv2Dataset(csv_file='data/hasyv2/hasy-data-labels.csv', root_dir='data/hasyv2/')

for i in range(len(hasy_dataset)):
    sample = hasy_dataset[i] 
    print(i, sample['image'].shape, sample['label'])
    showImage(sample['image'])

    if i == 3:
        break

#Test transform operations
scale = Hasyv2Dataset.Rescale(20)
crop = Hasyv2Dataset.RandomCrop(15)
composed = transforms.Compose([Hasyv2Dataset.Rescale(20),Hasyv2Dataset.RandomCrop(15)])


fig = plt.figure()
sample = hasy_dataset[0]
for i, tsfrm in enumerate([scale, crop, composed]):# Apply each of the above transforms on sample.
    transformed_sample = tsfrm(sample)

    showImage(transformed_sample['image'])

imgTensor = Hasyv2Dataset.ToTensor()(sample) #Test ToTensor operation
print(imgTensor['image'].shape)





