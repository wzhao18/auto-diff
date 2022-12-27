import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    seq = nn.Sequential(*[
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    ])
    res = nn.Residual(seq)
    relu = nn.ReLU()
    blk =  nn.Sequential(*[res, relu])
    return blk
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for _ in range(num_blocks):
        modules.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    modules.append(nn.Linear(hidden_dim, num_classes))
    seq = nn.Sequential(*modules)
    return seq
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train() 
    else:
        model.eval()

    iters, total_loss, num_corr = 0, 0, 0
    num_samples = len(dataloader.dataset)

    loss_fn = nn.SoftmaxLoss()
    flatten_fn = nn.Flatten()

    for inputs, labels in dataloader:
        inputs = flatten_fn(inputs)
        outputs = model(inputs)
        preds = np.argmax(outputs.numpy(), axis=1)

        loss = loss_fn(outputs, labels)
        total_loss += loss_fn(outputs, labels)
        num_corr += np.sum(preds == labels.numpy())
        iters += 1

        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()

    avg_loss = total_loss.numpy() / iters
    accu = num_corr / num_samples
    err = 1 - accu

    return err, avg_loss
    ### END YOUR SOLUTION


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_images_path = f"{data_dir}/train-images-idx3-ubyte.gz"
    train_labels_path = f"{data_dir}/train-labels-idx1-ubyte.gz"
    test_images_path = f"{data_dir}/t10k-images-idx3-ubyte.gz"
    test_labels_path = f"{data_dir}/t10k-labels-idx1-ubyte.gz"

    train_dataset = ndl.data.MNISTDataset(train_images_path, train_labels_path)
    test_dataset = ndl.data.MNISTDataset(test_images_path, test_labels_path)

    train_dataloader = ndl.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = MLPResNet(28 * 28, hidden_dim, num_classes=10)
    opt = optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train
    for i in range(epochs):
        train_err, train_loss = epoch(train_dataloader, model, opt)
        train_accu = 1 - train_err
        print(f"Epoch {i}: Average Loss: {train_loss}, Accuracy: {train_accu}")

    # Test
    test_err, test_loss = epoch(test_dataloader, model)
    test_accu = 1 - test_err
    print(f"Test: Average Loss: {test_loss}, Accuracy: {test_accu}")

    return train_accu, train_loss, test_accu, test_loss
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
