from tqdm.auto import tqdm
import torch

# train step
def train_step(model, dataloader, loss_fn, optimizer,
              device):

    model.train()

    train_loss, train_acc = 0,0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += ((y_pred_class== y).sum().item())/len(y_pred)

    train_loss = train_loss/len(dataloader)
    train_acc = train_acc/len(dataloader)
    return train_loss, train_acc

# test step
def test_step(model, dataloader, loss_fn, device):

    model.eval()

    test_loss, test_acc = 0,0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
    
            test_pred = model(X)
    
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
    
            test_pred_label = torch.argmax(torch.softmax(test_pred, dim=1), dim=1)
            test_acc += ((test_pred_label == y).sum().item())/len(test_pred_label)

    test_loss = test_loss/len(dataloader)
    test_acc = test_acc/len(dataloader)
    return test_loss, test_acc

# train loop
def train(model,
         train_dataloader,
         test_dataloader,
         optimizer,
         loss_fn,
         epochs,
         device):
    results = {'train_loss': [],
              'train_acc': [],
              'test_loss':[],
              'test_acc':[]}


    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model= model,
                                          dataloader= train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer = optimizer,
                                          device = device)
        test_loss, test_acc = test_step(model=model,
                                       dataloader= test_dataloader,
                                       loss_fn = loss_fn,
                                       device = device)

        print(
            f'epoch: {epoch+1} | '
            f'train loss: {train_loss:.4f} | '
            f'train acc: {train_acc:.4f} | '
            f'test loss: {test_loss:.4f} | '
            f'test acc: {test_acc:.4f} | '
        )

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

    return results