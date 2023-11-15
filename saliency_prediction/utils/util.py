def mean_std(test_list):
    mean = sum(test_list) / len(test_list) 
    variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list) 
    res = variance ** 0.5
    return mean , res 

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))