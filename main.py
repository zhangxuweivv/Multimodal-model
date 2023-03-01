import torch
from sklearn.metrics import classification_report
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch import nn, optim
from tqdm import tqdm

import data_loader
import Multimodalmodel #Multimodalmodel

def run_training():

    batch_size = 2
    learning_rate = 0.01
    num_epoches = 20
    momentum = 0.5

    data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    train_dataset = data_loader.Dataset(r'ecg_train_path', r'erf&be_train_path', transform=data_tf)
    validation_dataset = data_loader.Dataset(r'ecg_validations_path',r'erf&be_validations_path', transform=data_tf)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda:0")

    model = Multimodalmodel.Multimodalmodel()
    Net = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    losses = []
    acces = []

    for epoch in range(num_epoches):
        train_loss = 0
        train_acc = 0
        model.train()
        if epoch % 5 == 0:
            optimizer.param_groups[0]['lr'] *= 0.1
        for i, data in tqdm(enumerate(train_loader)):
            img,huayan,label = data[0].to(device), data[1].to(device), data[2].to(device)
            out = model(img,huayan)
            optimizer.zero_grad()
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            train_acc += acc
        losses.append(train_loss / len(train_loader))
        acces.append(train_acc / len(train_loader))
        print('epoch:{},Train Loss:{:.4f},Train Acc:{:.4f}'.format(epoch, train_loss / len(train_loader), train_acc / len(train_loader)))

        PATH = './MultimodalModel/MultimodalModel_net'
        torch.save(Net.state_dict(), PATH+str(epoch)+'.pth')

        model.eval()
        eval_loss = 0
        eval_acc = 0
        predlist = []
        labellist = []
        for i,data in enumerate(validation_loader):
            img, huayan, label = data[0].to(device), data[1].to(device), data[2].to(device)
            out = model(img,huayan)
            loss = criterion(out, label)
            eval_loss += loss.data.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.item()
            for i in range(pred.shape[0]):
                predlist.append(pred[i].cpu().numpy())
                labellist.append(label[i].cpu().numpy())
        report = classification_report(labellist, predlist, digits=4)
        print(report)
        print('Test Loss: {:.6f}, Acc: {:.6f}' .format(
            eval_loss / (len(validation_dataset)),
            eval_acc / (len(validation_dataset))
        ))

def run_test():
    batch_size = 2
    data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    test_dataset = data_loader.Dataset1('D:\\zxw\\reall\\teststrong', r'D:\\zxw\\reall\\ALLTest.xls',
                                              transform=data_tf)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0")
    model = Multimodalmodel.Multimodalmodel()

    save_dir = './save_path'#save_path
    # testmodel
    eval_acc = 0
    model.load_state_dict(torch.load(save_dir))
    print(model)
    model = model.to(device)
    model.eval()
    predlist = []
    labellist = []
    for i, data in enumerate(test_loader):
        img, huayan, label = data[0].to(device), data[1].to(device), data[2].to(device)
        out = model(img, huayan)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
        for i in range(pred.shape[0]):
            predlist.append(pred[i].cpu().numpy())
            labellist.append(label[i].cpu().numpy())
    report = classification_report(labellist, predlist, digits=4)
    print(report)
    print('Acc: {:.6f}'.format(
        eval_acc / (len(test_dataset))
    ))

def main():
    run_training()#train
    run_test()#test
if __name__ == '__main__':
    main()

