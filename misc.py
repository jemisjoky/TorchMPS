import numpy as np
import torch

def load_HV_data(length):
    """
    Output a toy "horizontal/vertical" data set of black and white 
    images with size length x length. Each image contains a single 
    horizontal or vertical stripe, set against a background
    of the opposite color. The labels associated with these images
    are either 0 (horizontal stripe) or 1 (vertical stripe).

    In its current version, this returns two data sets, a training
    set with 75% of the images and a test set with 25% of the 
    images.
    """
    num_images = 4 * (2**(length-1) - 1)
    num_patterns = num_images // 2
    split = num_images // 4

    if length > 14:
        print("load_HV_data will generate {} images, "
              "this could take a while...".format(num_images))

    images = np.empty([num_images,length,length], dtype=np.float32)
    labels = np.empty(num_images, dtype=np.int)

    # Used to generate the stripe pattern from integer i below
    template = "{:0" + str(length) + "b}"

    for i in range(1, num_patterns+1):
        pattern = template.format(i)
        pattern = [int(s) for s in pattern]

        for j, val in enumerate(pattern):
            # Horizontal stripe pattern
            images[2*i-2, j, :] = val
            # Vertical stripe pattern
            images[2*i-1, :, j] = val
        
        labels[2*i-2] = 0
        labels[2*i-1] = 1

    # Shuffle and partition into training and test sets
    np.random.seed(0)
    np.random.shuffle(images)
    np.random.seed(0)
    np.random.shuffle(labels)

    train_images, train_labels = images[split:], labels[split:]
    test_images, test_labels = images[:split], labels[:split]

    return torch.from_numpy(train_images), \
           torch.from_numpy(train_labels), \
           torch.from_numpy(test_images), \
           torch.from_numpy(test_labels)

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

def joint_shuffle(input_imgs, input_lbls):
    """
    Take pytorch arrays of images and labels, jointly shuffle
    them so that each label remains pointed to its corresponding
    image, then return the reshuffled tensors. Works for both
    regular and CUDA tensors.
    """
    assert input_imgs.is_cuda == input_lbls.is_cuda
    use_gpu = input_imgs.is_cuda
    if use_gpu:
        input_imgs, input_lbls = input_imgs.cpu(), input_lbls.cpu()

    images, labels = input_imgs.numpy(), input_lbls.numpy()

    # Shuffle relative to the same seed
    np.random.seed(0)
    np.random.shuffle(images)
    np.random.seed(0)
    np.random.shuffle(labels)

    images, labels = torch.from_numpy(images), torch.from_numpy(labels)
    if use_gpu:
        images, labels = images.cuda(), labels.cuda()

    return images, labels