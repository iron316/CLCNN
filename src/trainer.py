import torch


def train(model, train_loader, loss_func, device, optimizer):
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device).view(-1, 1)
        predict = model(x)
        loss = loss_func(predict, y)
        train_loss += loss.item()/len(train_loader)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss


def valid(model, train_loader, loss_func, device):
    model.eval()
    valid_loss = 0.0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device).view(-1, 1)
        with torch.no_grad():
            predict = model(x)
        loss = loss_func(predict, y)
        valid_loss += loss.item()/len(train_loader)
    return valid_loss
