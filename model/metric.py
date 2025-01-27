import torch
import torch.nn as nn

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def ccc(x, y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    sxy = torch.sum((x - x_mean) * (y - y_mean)) / x.shape[0]
    rhoc = (
        2
        * sxy
        / (
            torch.var(x, unbiased=False)
            + torch.var(y, unbiased=False)
            + (x_mean - y_mean) ** 2+1e-10
        )
    )
    return rhoc

def ccc_score(output, target):
    assert output.shape == target.shape
    batch_size, num_features = output.shape
    ccc_values = torch.zeros(batch_size)
    for i in range(batch_size):
        ccc_values[i] = ccc(output[i], target[i])
    mean_ccc = torch.mean(ccc_values)
    return mean_ccc

def pearson_corr(output, target):
    assert output.shape == target.shape
    true_mean = torch.mean(output, dim=1, keepdim=True)
    pred_mean = torch.mean(target, dim=1, keepdim=True)
    numerator = torch.sum((output - true_mean) * (target - pred_mean), dim=1)
    denominator = torch.sqrt(torch.sum((output - true_mean) ** 2, dim=1) * torch.sum((target - pred_mean) ** 2, dim=1))
    pcc_values = numerator / (denominator + 1e-10)
    return torch.mean(pcc_values)

def rmse(output, target):
    target = target.reshape(target.size(0),target.size(1))
    output = output.reshape(output.size(0),output.size(1))
    criterion = nn.MSELoss(reduction="mean")
    return torch.sqrt(criterion(output,target))
