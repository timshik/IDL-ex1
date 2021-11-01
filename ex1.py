from DatasetLoader import MyDataset, ProcessData, label_names
from model import SimpleModel
import model
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sn
import pandas as pd
import matplotlib.patches as mpatches
import sklearn.metrics as met

path_pos = 'pos_A0201.txt'
path_neg = 'neg_A0201.txt'
tr_batch_size = 16
test_batch_size = 32

mydata = ProcessData(path_pos, path_neg)
mydata.shuffle()
mydata.split_train_test()
trainset = MyDataset(mydata.get_train_data())
testset = MyDataset(mydata.get_test_data())
num_of_examples = len(trainset)
classes = list(label_names().values())

# this function will compute the distrbution of labels in the data
def count_labels(dataiter, num_of_examples=None):

    positive = 0
    negative = 0

    try:
        while(True):
            peptides, labels = dataiter.next()
            negative += labels.tolist().count(0)
            positive += labels.tolist().count(1)
    except:
        if num_of_examples is not None:
            print('of the labels are positive %.2f %%' % (positive*100/num_of_examples))
            print('of the labels are negative %.2f %%' % (negative*100/num_of_examples))



def get_weights(data, x=1.0, y=1.0):#x will represent the factor of the weight of the first class and y for the second
    weightsample = [0 for i in range(len(classes))]
    for i in range(len(trainset)):
        weightsample[int(trainset[i][1])] += 1

    weightsample = [1-(i/num_of_examples) for i in weightsample]
    weightsample[0] *= x
    weightsample[1] *= y

    weights = []
    for i in range(len(trainset)):
        weights.append(weightsample[trainset[i][1]])
    return weights


#loading the data to loaders
weights = get_weights(trainset, 1.3, 0.8)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(trainset), replacement=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=tr_batch_size, shuffle=False, sampler=sampler)#, sampler=sampler
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)
count_labels(iter(trainloader), num_of_examples)
count_labels(iter(testloader), len(testloader)*test_batch_size)

# train loop
#parameters: the model, train data,test data, loos function, optimizer, num of epochs
def train(model, trainloader, testloader, loss, optimizer, epochs):
    tr_losses = []
    dev_losses = []
    for ep in range(epochs):
        running_loss = 0
        model.train()

        for batch_idx, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            tr_loss = loss(torch.flatten(outputs), labels)
            tr_loss.backward()
            optimizer.step()
            running_loss += tr_loss.item()# sum of all batches

            if batch_idx % 1000 == 0:    # printing the loss of random batch
                print('[%d, %5d] loss: %.3f' %
                      (ep + 1, batch_idx + 1, tr_loss.item()))
                #evaluating on dev set
        model.eval()
        with torch.no_grad():
            running_dev_loss = 0
            for data in testloader:
                peptides, labels = data
                peptides = peptides.float()
                labels = labels.float()
                outputs = model(peptides)
                dev_loss = loss(torch.flatten(outputs), labels)
                running_dev_loss += dev_loss
            tr_losses.append(running_loss/len(trainloader)) #mean loss for the epoch
            dev_losses.append(running_dev_loss/len(testloader))
    return tr_losses, dev_losses


# actual train
model = SimpleModel()
loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00065, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3, amsgrad=False) #weight_decay=1e-5,
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, nesterov=False)
tr_losses, dev_losses = train(model, trainloader, testloader, loss, optimizer, 15)

# plottind the losses
# the dev bar might be below the train bar due to the diffrent distribution of the dev set
iteration = np.arange(0, len(tr_losses))
plt.plot(iteration, tr_losses, 'g-', iteration, dev_losses, 'r-')
plt.xlabel('iterations')
plt.ylabel('loss')
green_patch = mpatches.Patch(color='green', label='train')
red_patch = mpatches.Patch(color='red', label='test')
plt.legend(handles=[green_patch, red_patch])
plt.show()

#saving the model
path = './model_weights.ckpt'
model.save(path)

# evaluates the model with f1 score and confusion matrix
def evaluate(model,loader,train_or_test):
    if train_or_test == 'train':
        batch = tr_batch_size
    else:
        batch = test_batch_size
    model.eval()
    label_list = []
    predicted_list = []
    for data in loader:
        peptides, labels = data
        peptides = peptides.float()
        labels = labels.float()
        label_list.extend(labels.tolist())
        outputs = model(peptides)
        predicted = torch.round(outputs)
        predicted_list.extend(predicted.tolist())

    ### accuracy
    predicted_list_np = np.ndarray.flatten(np.array(predicted_list))
    label_list_np = np.ndarray.flatten(np.array(label_list))
    print('accuracy of the network %.2f %%' % (sum(predicted_list_np == label_list_np)/len(predicted_list)*100))

    ### f1_score
    f1_score = met.f1_score(label_list, predicted_list, average='macro')*100
    print(' f1_score of the network %s : %.2f %%' % (train_or_test,
                                                         f1_score))
    ### confusion matrix
    mat = met.confusion_matrix(label_list, predicted_list, labels=[0, 1])

    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4)# for label size
    df_cm = pd.DataFrame(mat, index=[i for i in classes], columns=[i for i in classes])
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 15})# font size

    plt.show()
    for i in range(len(classes)):
        print('  %s accuracy of %s : %d %%' % (classes[i], train_or_test,
                                                     mat[i][i]*100/(mat[i][0]+mat[i][1])))


evaluate(model, trainloader, 'train')
print('------------------------------------------')
evaluate(model, testloader, 'test')