import torch


def get_grads(named_parameters):
    record = {}
    for n, p in named_parameters:
        if p.requires_grad and "weight" in n:
            if p.grad is None:
                print(n," grad error: gradient returns None")
                raise AttributeError
            else:
                record[f"{n}_mean"] = p.grad.abs().mean().item()
                record[f"{n}_min"] = p.grad.abs().min().item()
                record[f"{n}_max"] = p.grad.abs().max().item()
    return record
