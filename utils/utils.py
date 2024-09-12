import os
from models.STSwinNet.load_pretrained import remap_pretrained_keys_swin,load_pretrained_interpolate
import mlflow
import pandas as pd
import torch
from collections.abc import MutableMapping


#for test or finetune
def load_model(prev_runid, model, device, remap = None, test=False):
    try:
        run = mlflow.get_run(prev_runid)
    except:
        return model

    model_dir = run.info.artifact_uri + "/model/data/model.pth"
    if model_dir[:7] == "file://":
        model_dir = model_dir[7:]

    if os.path.isfile(model_dir):
        pretrained_model = torch.load(model_dir, map_location=device)
        #model.load_state_dict(model_loaded.state_dict())
        #for data parallel model
        pretrained_dict = pretrained_model.state_dict()
        if test:
            pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
        if remap == "v2":
            print(">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
            pretrained_dict = remap_pretrained_keys_swin(model, pretrained_dict)
            del pretrained_model
            torch.cuda.empty_cache()
        elif remap == "v1":
            load_pretrained_interpolate(model,pretrained_dict)
            del pretrained_model
            torch.cuda.empty_cache()
        model.load_state_dict(pretrained_dict, strict=False)
        print("Model restored from " + prev_runid + "\n")
    else:
        print("No model found at" + prev_runid + "\n")

    return model

def resume_model(prev_runid, optimizer, scheduler, scaler, epoch_initial, device):

    run = mlflow.get_run(prev_runid)


    state_dir = run.info.artifact_uri + "/training_state_dict/state_dict.pth"
    if state_dir[:7] == "file://":
        state_dir = state_dir[7:]

    if os.path.isfile(state_dir):

        state_dict = torch.load(state_dir, map_location=device)
        # for item in state_dict["optimizer"]["state"]:
        #     print(state_dict["optimizer"]["state"][item]["exp_avg"].shape)
        if "optimizer" in state_dict.keys():
            optimizer.load_state_dict(state_dict["optimizer"])
        if "scheduler" in state_dict.keys() and scheduler is not None:
            scheduler.load_state_dict(state_dict["scheduler"])
        if "scaler" in state_dict.keys() and scaler is not None:
            scaler.load_state_dict(state_dict["scaler"])
        epoch_initial = state_dict["epoch"] + 1

        print("Model resumed from " + prev_runid + "\n")
    else:
        print("No model found at" + prev_runid + "\n")

    #resume previous metrics
    # for key, value in run.data.metrics.items():
    #     mlflow.log_metric(key, value)
    # train_loss_file = os.path.dirname( run.info.artifact_uri) + "/metrics/train_loss"
    # valid_loss_file = os.path.dirname( run.info.artifact_uri) + "/metrics/valid_loss"
    # if os.path.isfile(train_loss_file):
    #     with open(train_loss_file, 'r') as f:
    #         train_loss = f.read()
    #         mlflow.log_metric("train_loss", float(train_loss))
    # if os.path.isfile(train_loss_file):
    #     with open(valid_loss_file, 'r') as f:
    #         valid_loss = f.read()
    #         mlflow.log_metric("valid_loss", float(valid_loss))

    return optimizer, scheduler, scaler, epoch_initial

def create_model_dir(path_results, runid):
    path_results += runid + "/"
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    print("Results stored at " + path_results + "\n")
    return path_results


def save_model(model):
    mlflow.pytorch.log_model(model, "model")


def save_state_dict(optimizer,scheduler,scaler, epoch):
    state_dict = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "scaler": scaler.state_dict() if scaler else None,
    }
    mlflow.pytorch.log_state_dict(state_dict, artifact_path="training_state_dict")


def save_csv(data, fname):
    # create file if not there
    path = mlflow.get_artifact_uri(artifact_path=fname)
    if path[:7] == "file://":  # to_csv() doesn't work with 'file://'
        path = path[7:]
    if not os.path.isfile(path):
        mlflow.log_text("", fname)
        pd.DataFrame(data).to_csv(path)
    # else append
    else:
        pd.DataFrame(data).to_csv(path, mode="a", header=False)


def save_flops_csv(data, fname):
    # create file if not there
    path = mlflow.get_artifact_uri(artifact_path=fname)
    if path[:7] == "file://":  # to_csv() doesn't work with 'file://'
        path = path[7:]
        mlflow.log_text("", fname)
        data = flatten_dict(data)
        df = pd.DataFrame.from_dict(data, orient='index', columns=['flops'])
        df.to_csv(path)
    # else append
    # else:
    #     pd.DataFrame(data).to_csv(path, mode="a", header=False)

def save_diff(fname="git_diff.txt"):
    # .txt to allow showing in mlflow
    path = mlflow.get_artifact_uri(artifact_path=fname)
    if path[:7] == "file://":
        path = path[7:]
    mlflow.log_text("", fname)
    os.system(f"git diff > {path}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape,  param.device)
        # if torch.isnan(param):
        #     print("Nan value:", name)

    return 0






def flatten_dict(d: MutableMapping, sep: str= '.') -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict

