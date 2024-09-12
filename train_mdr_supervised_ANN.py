import argparse
import mlflow
import torch
from torch.optim import *
#from torchsummary import summary
from configs.parser import YAMLParser
from loss.flow_supervised import *
from tqdm import tqdm
import math
from models.STSwinNet.STSwinNet import STTFlowNet, STTFlowNet_4en
from utils.gradients import get_grads
from utils.utils import load_model, save_csv, save_model, save_state_dict, resume_model, count_parameters,print_parameters
from utils.visualization import Visualization_DSEC
# from utils.print_gpu_info import print_gpu_utilization,print_gpu_summary
from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite
from DSEC_dataloader.data_augmentation import downsample_data,Compose,CenterCrop,RandomCrop,RandomRotationFlip,Random_event_drop,Random_horizontal_flip,Random_vertical_flip
from utils.mlflow import log_config, log_results
import torch.nn.functional as F
import cv2
from torch.utils.data.dataloader import DataLoader
from MDR_dataloader.MDR import MDREventFlow

import random

use_ml_flow = True


def train(args, config_parser):
    ########## configs ##########
    config = config_parser.config
    # initialize settings
    device = config_parser.device
    print('device:', device)
    if device.type == 'cpu':
        config["loader"]["batch_size"] = 1
        config["loader"]["n_workers"] = 0
        config["loader"]["gpu"] = None

    # log config
    if use_ml_flow:
        mlflow.set_tracking_uri(args.path_mlflow)
        mlflow.set_experiment(config["experiment"])
        mlflow.start_run()
        mlflow.log_params(config)
        mlflow.log_param("prev_runid", args.prev_runid)
        print("MLflow dir:", mlflow.active_run().info.artifact_uri[:-9])

    config = config_parser.combine_entries(config)

    #use mix-precision training
    if config['optimizer']['use_amp']:
        print("[warning] using torch.cuda.amp.GradScaler()")
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # visualization tool
    if config["vis"]["enabled"]:
        vis = Visualization_DSEC(config)

    ########## data loader ##########

    # Define data augmentation


    if config['loader']['crop'] is not None:
        config["swin_transformer"]["input_size"] = [config['loader']['crop'][0], config['loader']['crop'][1]]
    else:
        config["swin_transformer"]["input_size"] = [config['loader']['resolution'][0], config['loader']['resolution'][1]]

    # Create training dataset
    print("Training Dataset ...")


    train_dataset = MDREventFlow(
        config = config,
        train=True,
        aug=True,
    )

    if(config["data"]["event_interval"] == "dt1"):
        from MDR_dataloader.MVSEC import MvsecEventFlow
        valid_dataset = MvsecEventFlow(
            config= config,
            train=False,
            aug=False
        )
    elif(config["data"]["event_interval"]  == "dt4"):
        from MDR_dataloader.MVSEC import MvsecEventFlow_dt4
        valid_dataset = MvsecEventFlow_dt4(
            config = config,
            train=False,
            aug=False
        )
    else:
         raise Exception('Please provide a valid input setting (dt1 or dt4)!')


    # Instantiate Dataloader
    train_dataloader = DataLoader(train_dataset,
                                batch_size=config["loader"]["batch_size"],
                                shuffle=True,
                                num_workers=config["loader"]["n_workers"],
                                pin_memory=True,
                                drop_last=True)
    valid_dataloader = DataLoader(valid_dataset,
                                 batch_size=config["loader"]["batch_size"],
                                 shuffle=False,
                                 num_workers=config["loader"]["n_workers"],
                                 pin_memory=True,
                                 drop_last=True)

    ############## Training ###############
    # model initialization and settings
    #Transformer config
    if config["swin_transformer"]["use_arc"][0]:
        model = eval(config["model"]["name"])(config["model"].copy(), config["swin_transformer"].copy())
    else:
        model = eval(config["model"]["name"])(config["model"].copy())


    model.to(device)

    #multi-gpu initial weights
    if type(config["loader"]["gpu"]) == str:
        device_ids = list(map(int, config["loader"]["gpu"].split(',')))
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.module.init_weights()
    else:
        model.init_weights()

    epoch_initial = 0
    remap = None

    if args.finetune:

        if config["swin_transformer"]["use_arc"][0] == "swinv2":
            remap = "v2"
        elif config["swin_transformer"]["use_arc"][0] == "swinv1":
            remap = "v1"

    model = load_model(args.prev_runid, model, device, remap)

    #summary(model)
    print(model)
    print_parameters(model)
    print("Total parameters: ", count_parameters(model))

    # log config
    if use_ml_flow:
        mlflow.log_param("number of params", count_parameters(model))

    # optimizers
    if config["optimizer"]["name"] == 'AdamW':
        optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"],weight_decay=config["optimizer"]["wd"])
    else:
        optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"])


    if config["optimizer"]["scheduler"] == "cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["optimizer"]["lr"],
            total_steps=config["loader"]["n_epochs"] + 100,
            pct_start=0.01,
            cycle_momentum=False,
            anneal_strategy='linear')
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["optimizer"]["milestones"],
                                                         gamma=0.5)
    optimizer.zero_grad()
    # if config["optimizer"]["num_acc"]:
    #     num_acc_steps = config["optimizer"]["num_acc"]
    # else:
    num_acc_steps = 1.
    if args.resume:
        optimizer, scheduler, scaler, epoch_initial = resume_model(args.prev_runid, optimizer, scheduler, scaler, epoch_initial, device)

    # Define the loss function
    loss_function = flow_loss_supervised(config,device)

    # simulation variables

    best_loss = 1.0e6
    grads_w = []

    # training loop

    for epoch in range(epoch_initial, config["loader"]["n_epochs"]):
        print(f'Epoch {epoch}')
        model.train()
        sample = 0
        train_loss = 0.


        for data in tqdm(train_dataloader):
            torch.autograd.set_detect_anomaly(True)
            chunk = data['d_event_volume_new'].to(device=device, dtype=torch.float32)
            if config["data"]["num_chunks"] == 2:
                chunk_old = data['d_event_volume_old'].to(device=device, dtype=torch.float32)
                chunk = torch.cat((chunk_old, chunk), dim=1)
            label = data['flow'].to(device=device, dtype=torch.float32)
            mask = data['valid'].unsqueeze(dim=1).to(device=device, dtype=torch.float32)


            with torch.cuda.amp.autocast(enabled=scaler is not None):
                # forward pass

                if config['model']['encoding'] == 'cnt':
                    if config["vis"]["enabled"]:
                        chunk_vis = torch.sum(chunk, dim=1)

                    else:
                        if config['loader']['polarity']:
                            chunk = chunk.view([chunk.shape[0], -1] + list(chunk.shape[3:]))#for EV-FlowNet  #[B,40,H,W] 2+2+2....

                    pred_list = model(None, chunk.to(device))

                elif config['model']['encoding'] == 'voxel':
                    neg = torch.nn.functional.relu(-chunk)
                    pos = torch.nn.functional.relu(chunk)
                    # ignore polarity
                    if not config['loader']['polarity']:
                        chunk = torch.cat((torch.unsqueeze(pos,dim=2), torch.unsqueeze(neg,dim=2)), dim=2)
                    if config["vis"]["enabled"]:
                        chunk_vis = torch.stack((torch.sum(pos, dim=1), torch.sum(neg, dim=1)), dim=1)

                    pred_list = model(chunk.to(device), None)


                else:
                    print("Config error: Event encoding not support.")
                    raise AttributeError

                #backward pass
                if config["metrics"]["mask_events"]:
                    event_mask = torch.unsqueeze(torch.sum(chunk, dim=1).bool(), dim=1)
                    curr_loss = loss_function(pred_list["flow"], label, mask*event_mask, gamma = config["loss"]["gamma"]) / num_acc_steps
                else:
                    curr_loss = loss_function(pred_list["flow"], label, mask, gamma = config["loss"]["gamma"])/num_acc_steps

                if np.isnan(curr_loss.item()):
                    raise

            if scaler is not None:
                scaler.scale(curr_loss).backward()
            else:
                curr_loss.backward()



            # clip and save grads

            if config["loss"]["clip_grad"] is not None:
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
            if config["vis"]["store_grads"]:
                grads_w.append(get_grads(model.named_parameters()))

            if ((sample + 1) % num_acc_steps == 0) or (sample + 1 == len(train_dataloader)):
                # Update Optimizer
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()



            train_loss += curr_loss.item() * config["loader"]["batch_size"]



            # visualize
            if config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                # mask flow for visualization
                flow_vis = pred_list["flow"][-1].clone()
                # flow_vis *= mask
                if config["vis"]["mask_events"]:
                    event_mask = torch.unsqueeze(torch.sum(chunk, dim=1).bool(), dim=1)
                    flow_vis *= event_mask

                with torch.no_grad():
                    vis.update(chunk_vis, label, mask, flow_vis, None)

            # print training info
            sample += 1





        # save grads to file
        if config["vis"]["store_grads"]:
            save_csv(grads_w, "grads_w.csv")
            grads_w = []

        epoch_loss = train_loss / sample
        print(f'Epoch loss = {epoch_loss}')
        if use_ml_flow:
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)

        # save model
        with torch.no_grad():
            if epoch_loss < best_loss:
                save_model(model)
                save_state_dict(optimizer, scheduler, scaler, epoch)
                best_loss = epoch_loss


        #####validate after each 5 epoch############
        # Validation Dataset

        if epoch % config["test"]["n_valid"] == 0:
            sample = 0
            model.eval()
            epoch_loss_valid = 0.
            print('Validating... (test sequence)')

            #desactivate autograd
            with torch.set_grad_enabled(False):
                for data in tqdm(valid_dataloader):
                    torch.autograd.set_detect_anomaly(True)
                    chunk = data['event_volume_new'].to(device=device, dtype=torch.float32)
                    if config["data"]["num_chunks"] == 2:
                        chunk_old = data['event_volume_old'].to(device=device, dtype=torch.float32)
                        chunk = torch.cat((chunk_old, chunk), dim=1)
                    label = data['flow'].to(device=device, dtype=torch.float32)

                    with torch.no_grad():

                        # forward pass
                        if config['model']['encoding'] == 'cnt':
                            if config["vis"]["enabled"]:
                                chunk_vis = torch.sum(chunk, dim=1)
                            if config["swin_transformer"]["use_arc"][1] == "PatchEmbed3D":
                                chunk = torch.transpose(chunk, 1, 2)
                            else:
                                if config['loader']['polarity']:
                                    chunk = chunk.view([chunk.shape[0], -1] + list(
                                        chunk.shape[3:]))  # for EV-FlowNet  #[B,40,H,W] 2+2+2....
                            pred_list = model(None, chunk.to(device))
                        elif config['model']['encoding'] == 'voxel':
                            neg = torch.nn.functional.relu(-chunk)
                            pos = torch.nn.functional.relu(chunk)
                            # polarity in additional channel
                            if not config['loader']['polarity']:
                                chunk = torch.cat((torch.unsqueeze(pos, dim=2), torch.unsqueeze(neg, dim=2)), dim=2)

                            if config["vis"]["enabled"]:
                                chunk_vis = torch.stack((torch.sum(pos, dim=1), torch.sum(neg, dim=1)), dim=1)

                            pred_list = model(chunk.to(device), None)
                        else:
                            print("Config error: Event encoding not support.")
                            raise AttributeError

                        pred = pred_list["flow"][-1]

                        if config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                            # mask flow for visualization
                            flow_vis = pred_list["flow"][-1].clone()
                            # flow_vis *= mask
                            if config["vis"]["mask_events"]:
                                event_mask = torch.unsqueeze(torch.sum(chunk, dim=1).bool(), dim=1)
                                flow_vis *= event_mask

                                vis.update(chunk_vis, label, mask, flow_vis, None)

                    # backward pass
                    if config["metrics"]["mask_events"]:
                        event_mask = torch.unsqueeze(torch.sum(chunk, dim=1).bool(), dim=1)
                        total_loss = loss_function([pred], label, mask * event_mask)
                    else:
                        total_loss = loss_function([pred], label, mask)

                    epoch_loss_valid += total_loss.item() * config["loader"]["batch_size"]
                    sample += 1


            epoch_loss_valid = epoch_loss_valid/sample
            print('Epoch loss (Validation): {} \n'.format(epoch_loss_valid))
            if use_ml_flow:
                mlflow.log_metric("valid_loss", epoch_loss_valid, step=epoch)

        # update learning rate
        if scheduler is not None:
            scheduler.step()



    if use_ml_flow:
        mlflow.end_run()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train_MDR_supervised_STT_voxel.yml",
        help="training configuration",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    parser.add_argument(
        "--prev_runid",
        default="",
        help="pre-trained model to use as starting point",
    )
    parser.add_argument(
        "--save_path",
        default="results/checkpoint_epoch{}.pth",
        help="save the model",
    )


    parser.add_argument(
        "--finetune",
        default="",
        help="fine tune the model on full resolution",
    )

    parser.add_argument(
        "--resume",
        default="",
        help="resume the training",
    )
    args = parser.parse_args()

    # launch training
    train(args, YAMLParser(args.config))



