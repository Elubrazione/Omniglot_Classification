import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from arch_layers import ConvNet
from util_data import DataSet, data_transform
from draw_figures import draw_fig
from torch.utils.data import DataLoader

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# DEVICE = torch.device('cpu')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data.float())
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    # if(batch_idx+1)%5 == 0:
    #   print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
    #     epoch, batch_idx * len(data), len(train_loader.dataset),
    #     100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, batch):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      one_hot_index = list(np.arange(5))
      one_hot_index = np.identity(5)[one_hot_index]

      data, target = data.to(device), target.to(device)
      output = model(data.float())
      test_loss += F.cross_entropy(output, target, reduction='sum') # 将一批的损失相加
      _, predicted = torch.max(output, 1)
      pred = []

      # TRANS TO ONE-HOT LABEL
      predicted = np.array(predicted)
      for item in predicted:
        pred.append(one_hot_index[item])
      pred = torch.from_numpy(np.array(pred, dtype=np.float))

      # CALCULATE ACCURACY
      for i in range(target.shape[0]):
        if (pred[i]==target[i]).sum().item() == 5:
          correct += 1

  test_loss /= len(test_loader.dataset)
  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  return correct / len(test_loader.dataset)


if __name__ == '__main__':
  if not os.path.exists('./figures'):
    os.makedirs('./figures')
  if not os.path.exists('./model'):
    os.makedirs('./model')
  if not os.path.exists('./results'):
    os.makedirs('./results')

  data_set = DataSet(5, 5, 5, re_size=True, transform=data_transform)
  train_dataset = data_set.get_train_data()
  test_dataset = data_set.get_test_data()
  batchs = 25
  epochs = 100
  learning_rate = 0.0005

  batch_list = [1, 5, 10, 15, 20, 25]
  # BATCHS INFLUENCE
  for batch in batch_list:
    print(f'BATCH: {batch}')
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True)
    TTL = 20
    # EPOCHS INFLUENCE
    test_accuracy = []
    model = ConvNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    acc_last = 0
    for epoch in range(1, epochs + 1):
      train(model, DEVICE, train_loader, optimizer, epoch)
      accuracy = test(model, DEVICE, test_loader, batch)
      test_accuracy.append(accuracy)

      if accuracy < acc_last:  TTL -= 1
      acc_last = accuracy
      with open(f'./results/lr{learning_rate}_batch{batch}_eval.txt', "a") as f:
        f.write(str(epoch) + '/' + str(accuracy) + '\n')
      if TTL <= 0: break
    torch.save(model, f'./model/lr{learning_rate}_model_net_'+str(batch)+'.pkl')
    draw_fig(test_accuracy, batch, learning_rate)