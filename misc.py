import numpy as np
import torch

def _invert(image):
    """
    Take a black and white image (black=0, white=1) and invert
    all the pixels
    """
    return image < 0.5

def load_HV_data(length):
    """
    Output a toy "horizontal/vertical" data set of size
    length x length black and white images. Each image contains
    a single horizontal or vertical stripe, set against a
    background of the opposite color.

    This also ouputs a collection of labels, which are either
    0 (horizontal stripe) or 1 (vertical stripe).
    """
    images = np.ones([4*length, length, length])
    labels = np.empty(4*length, dtype=np.int)

    for i in range(length):
        # Horizontal stripe at i'th column
        images[4*i,i,:] = 0
        images[4*i+1] = _invert(images[4*i])
        labels[4*i], labels[4*i+1] = 0, 0
        
        # Vertical stripe at i'th column
        images[4*i+2,:,i] = 0
        images[4*i+3] = _invert(images[4*i+2])
        labels[4*i+2], labels[4*i+3] = 1, 1

    return torch.from_numpy(images), torch.from_numpy(labels)

def convert_to_onehot(batch_labels, num_labels):
    """
    Take a list of discrete labels from the set 
    {0,1,...,num_labels-1} and return a corresponding batch of
    one-hot encoded vectors
    """
    if max(batch_labels) >= num_labels:
        raise ValueError("Label values larger than allowed maximum")

    label_tensor = torch.zeros([len(batch_labels), num_labels])
    for i, label in enumerate(batch_labels):
        label_tensor[i, label] = 1

    return label_tensor