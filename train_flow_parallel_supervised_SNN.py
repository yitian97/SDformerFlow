import argparse
import mlflow
import os
import torch
from torch.optim import *
from configs.parser import YAMLParser
#from utils.utils import print_parameters
from loss.flow_supervised import *
from tqdm import tqdm
import math
from models.STSwinNet_SNN.Spiking_STSwinNet import SpikingformerFlowNet, MS_SpikingformerFlowNet, MS_SpikingformerFlowNet_en4
from utils.gradients import get_grads
from utils.utils import load_model, save_csv, save_model, save_state_dict, resume_model, count_parameters,print_parameters,save_flops_csv
from utils.visualization import Visualization_DSEC
from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite
from DSEC_dataloader.data_augmentation import downsample_data,Compose,CenterCrop,RandomCrop,RandomRotationFlip,Random_event_drop,Random_horizontal_flip,Random_vertical_flip
from utils.mlflow import log_config, log_results
import torch.nn.functional as F
import cv2
import random
from spikingjelly.activation_based import functional,neuron, surrogate
from models.STSwinNet_SNN.Spiking_submodules import *
import collections
from torch.utils.data.distributed import DistributedSampler


use_ml_flow = True


def train(args, config_parser):
    ########## configs ##########
    config = config_parser.config
    # initialize settings
    device = config_parser.device
    print('device:', device)

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


    ############## Training ###############
    # model initialization and settings
    #Transformer config
    if config['loader']['crop'] is not None:
        config["swin_transformer"]["input_size"] = [config['loader']['crop'][0], config['loader']['crop'][1]]
    else:
        config["swin_transformer"]["input_size"] = [config['loader']['resolution'][0], config['loader']['resolution'][1]]

    if config["swin_transformer"]["use_arc"][0]:
        model = eval(config["model"]["name"])(config["model"].copy(), config["swin_transformer"].copy())
    else:
        model = eval(config["model"]["name"])(config["model"].copy())

    model.to(device)
    model.init_weights()
    #multi-gpu
    if type(config["loader"]["gpu"]) == str:
        device_ids = list(map(int, config["loader"]["gpu"].split(',')))
        model.module.init_weights()


    epoch_initial = 0
    remap = None

    if args.finetune:

        if config["swin_transformer"]["use_arc"][0] == "swinv2":
            remap = "v2"
        elif config["swin_transformer"]["use_arc"][0] == "swinv1":
            remap = "v1"

    model = load_model(args.prev_runid, model, device, remap)

    #reset SNN, step model, backend
    # tune the surrogate function
    if 'SG_alpha' in config['optimizer']:
        for m in model.modules():
            if isinstance(m, surrogate.ATan):
                m.alpha = config['optimizer']['SG_alpha']

    functional.reset_net(model)
    functional.set_step_mode(model, config['data']['step_mode'])


    if config["model"]["spiking_neuron"]["neuron_type"] == "if":
        neurontype = getattr(neuron, "IFNode")
    elif config["model"]["spiking_neuron"]["neuron_type"] == "lif":
        neurontype = getattr(neuron, "LIFNode")
    elif config["model"]["spiking_neuron"]["neuron_type"] == "plif":
        neurontype = getattr(neuron, "ParametricLIFNode")
    elif config["model"]["spiking_neuron"]["neuron_type"] == "glif":
        neurontype = GatedLIFNode
    elif config["model"]["spiking_neuron"]["neuron_type"] == "psn":
        neurontype = PSN
    elif config["model"]["spiking_neuron"]["neuron_type"] == "SLTTlif":
        neurontype = SLTTLIFNode
    else:
        raise "neurontype not implemented!"

    if device.type != 'cpu':
        functional.set_backend(model, "cupy", neurontype)

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

    if config["optimizer"]["scheduler"] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["optimizer"]["milestones"], gamma=0.5)
    else:
        scheduler = None
    optimizer.zero_grad()
    if config["optimizer"]["num_acc"] is not None:
        num_acc_steps = config["optimizer"]["num_acc"]
    else:
        num_acc_steps = 1.

    if args.resume:
        optimizer, scheduler, scaler, epoch_initial = resume_model(args.prev_runid, optimizer, scheduler, scaler, epoch_initial, device)

    # Define the loss function
    loss_function = flow_loss_supervised(config,device)

    ########## data loader ##########

    # Define data augmentation
    if args.finetune:
        transform_train = Compose([
            # RandomRotationFlip((0,0),config['loader']['augment_prob'][0],config['loader']['augment_prob'][1]),
            Random_horizontal_flip(config['loader']['augment_prob'][0]),
            Random_vertical_flip(config['loader']['augment_prob'][1]),
            # Random_event_drop(max_drop_rate=config['loader']['max_drop_rate'])
        ])
        transform_valid = None
    else:
        transform_train = Compose([
            # RandomRotationFlip((0,0),config['loader']['augment_prob'][0],config['loader']['augment_prob'][1]),
            RandomCrop((config['loader']['crop'][0],config['loader']['crop'][1])),
            Random_horizontal_flip(config['loader']['augment_prob'][0]),
            Random_vertical_flip(config['loader']['augment_prob'][1]),
            # Random_event_drop(max_drop_rate=config['loader']['max_drop_rate'])
        ])

        transform_valid = Compose([CenterCrop((config['loader']['crop'][0], config['loader']['crop'][1]))])




    # Create training dataset
    print("Training Dataset ...")
    train_dataset = DSECDatasetLite(
        config,
        file_list='train',
        stereo=False,
    )

    # Define training dataloader

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config["loader"]["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=config["loader"]["n_workers"]
    )


    # Create validation dataset centrer crop no scale
    print("Validation Dataset ...")
    valid_dataset = DSECDatasetLite(
        config,
        file_list='valid',
        stereo=False,
    )

    # Define validation dataloader

    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=config["loader"]["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=config["loader"]["n_workers"]
    )



    # simulation variables

    best_loss = 1.0e6
    grads_w = []

    #record gradients
    # input_grad_monitor = monitor.GradInputMonitor(model, neurontype, function_on_grad_input=torch.norm)

    # training loop

    for epoch in range(epoch_initial, config["loader"]["n_epochs"]):
        print(f'Epoch {epoch}')
        model.train()
        sample = 0
        train_loss = 0.
        # spiking_rates = collections.defaultdict(list)
        for chunk, mask, label in tqdm(train_dataloader):
            torch.autograd.set_detect_anomaly(True)

            functional.reset_net(model)
            functional.set_step_mode(model, config['data']['step_mode']) #layer-by-layer

            chunk = chunk.to(device=device, dtype=torch.float32) #[B,20,2,H,W]
            label = label.to(device=device, dtype=torch.float32)  # [num_batches, 2, H, W]
            mask = mask.to(device=device)
            mask = torch.unsqueeze(mask, dim=1)
            if transform_train is not None:
                chunk, label, mask = transform_train((chunk, label, mask.float()))

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                # forward pass
                if config['model']['encoding'] == 'cnt':
                    if config["vis"]["enabled"]:
                        chunk_vis = torch.sum(chunk, dim=1)
                    else:
                        if config['loader']['polarity']:
                            chunk = chunk.view([chunk.shape[0], -1] + list(chunk.shape[3:]))#for EV-FlowNet  #[B,40,H,W] 2+2+2....


                elif config['model']['encoding'] == 'voxel':
                    if config['loader']['polarity']:
                        # ignore polarity
                        neg = torch.nn.functional.relu(-chunk)
                        pos = torch.nn.functional.relu(chunk)
                        # chunk = torch.abs(chunk)
                        chunk = torch.cat((torch.unsqueeze(pos, dim=2), torch.unsqueeze(neg, dim=2)),
                                          dim=2)  # B,C=20,P=2,H,W   B C, P, H, W
                        if config["vis"]["enabled"]:
                            chunk_vis = torch.stack((torch.sum(pos, dim=1), torch.sum(neg, dim=1)), dim=1)
                    else:
                        if config["vis"]["enabled"]:
                            chunk_vis = torch.sum(chunk, dim=1).detach()


                else:
                    print("Config error: Event encoding not support.")
                    raise AttributeError

                # normalize input
                if config["model"]["norm_input"] == "minmax":
                    min, max = (
                        torch.min(chunk[chunk != 0]),
                        torch.max(chunk[chunk != 0]),
                    )
                    if not min == max:
                        chunk[chunk != 0] = (chunk[chunk != 0] - min) / (max - min)
                elif config["model"]["norm_input"] == "std":
                    mean, stddev = (
                        chunk[chunk != 0].mean(),
                        chunk[chunk != 0].std(),
                    )
                    if stddev > 0:
                        chunk[chunk != 0] = (chunk[chunk != 0] - mean) / stddev

                # spike input

                if config['data']['spike_th'] is not None:
                    chunk[chunk > config['data']['spike_th']] = 1
                    chunk[chunk < config['data']['spike_th']] = 0

                pred_list = model(chunk.to(device))
                pred = pred_list["flow"]

                #backward pass only the last flow pred
                if config["metrics"]["mask_events"]:
                    # event_mask = torch.unsqueeze(torch.sum(chunk, dim=1).bool(), dim=1)
                    event_mask = torch.sum(torch.sum(chunk, dim=1),dim=1, keepdim=True).bool()
                    curr_loss = loss_function(pred, label, mask*event_mask, gamma = config["loss"]["gamma"])/num_acc_steps
                else:
                    curr_loss = loss_function(pred, label, mask, gamma = config["loss"]["gamma"])/num_acc_steps
                # print("loss: ", curr_loss.item())

                if np.isnan(curr_loss.item()):
                    raise

            if scaler is not None:
                scaler.scale(curr_loss).backward()
            else:
                curr_loss.backward()

                # print(f' input_grad_monitor.records=\n{input_grad_monitor.records}\n')

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

                # zero grad
                optimizer.zero_grad()

            train_loss += curr_loss.item() * config["loader"]["batch_size"]

            # visualize
            if config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                # mask flow for visualization
                flow_vis = pred_list["flow"][-1].clone()
                # flow_vis *= mask
                if config["vis"]["mask_events"]:
                    event_mask = torch.sum(torch.sum(chunk, dim=1),dim=1, keepdim=True).bool()
                    flow_vis *= event_mask

                with torch.no_grad():
                    vis.update(chunk_vis, label, mask, flow_vis, None)

            # print training info
            sample += 1



        # save grads to file
        if config["vis"]["store_grads"]:
            save_csv(grads_w, "grads_w.csv")
            print("Grads saved to grads_w.csv")
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
            if config["loader"]["batch_size"] > 1:
                model.eval()
            else:
                model.train()
            epoch_loss_valid = 0.
            print('Validating... (test sequence)')

            #desactivate autograd
            with torch.set_grad_enabled(False):
                for chunk, mask, label in tqdm(valid_dataloader):

                    functional.reset_net(model)
                    functional.set_step_mode(model, config['data']['step_mode'])

                    chunk = chunk.to(device=device, dtype=torch.float32)
                    label = label.to(device=device, dtype=torch.float32)  # [num_batches, 2, H, W]
                    mask = mask.bool().to(device=device)
                    mask = torch.unsqueeze(mask, dim=1)

                    with torch.no_grad():

                        if transform_valid is not None:
                            chunk, label, mask = transform_valid((chunk, label, mask.float()))
                        # forward pass
                        if config['model']['encoding'] == 'cnt':
                            if config["vis"]["enabled"]:
                                chunk_vis = torch.sum(chunk, dim=1)
                            else:
                                if config['loader']['polarity']:
                                    chunk = chunk.view([chunk.shape[0], -1] + list(
                                        chunk.shape[3:]))  # for EV-FlowNet  #[B,40,H,W] 2+2+2....


                        elif config['model']['encoding'] == 'voxel':
                            if config['loader']['polarity']:
                                # ignore polarity
                                neg = torch.nn.functional.relu(-chunk)
                                pos = torch.nn.functional.relu(chunk)
                                # chunk = torch.abs(chunk)
                                chunk = torch.cat((torch.unsqueeze(pos, dim=2), torch.unsqueeze(neg, dim=2)),
                                                  dim=2)  # B,C=20,P=2,H,W   B C, P, H, W
                                if config["vis"]["enabled"]:
                                    chunk_vis = torch.stack((torch.sum(pos, dim=1), torch.sum(neg, dim=1)), dim=1)
                            else:
                                if config["vis"]["enabled"]:
                                    chunk_vis = torch.sum(chunk, dim=1).detach()


                        else:
                            print("Config error: Event encoding not support.")
                            raise AttributeError

                        # normalize input
                        if config["model"]["norm_input"] == "minmax":
                            min, max = (
                                torch.min(chunk[chunk != 0]),
                                torch.max(chunk[chunk != 0]),
                            )
                            if not min == max:
                                chunk[chunk != 0] = (chunk[chunk != 0] - min) / (max - min)
                        elif config["model"]["norm_input"] == "std":
                            mean, stddev = (
                                chunk[chunk != 0].mean(),
                                chunk[chunk != 0].std(),
                            )
                            if stddev > 0:
                                chunk[chunk != 0] = (chunk[chunk != 0] - mean) / stddev

                        # spike input
                        if config['data']['spike_th'] is not None:
                            chunk[chunk > config['data']['spike_th']] = 1
                            chunk[chunk < config['data']['spike_th']] = 0

                        pred_list = model(chunk.to(device))
                        pred = pred_list["flow"][-1]

                        if config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                            # mask flow for visualization
                            flow_vis = pred_list["flow"][-1].clone()
                            # flow_vis *= mask
                            if config["vis"]["mask_events"]:
                                # event_mask = torch.unsqueeze(torch.sum(torch.sum(chunk, dim=1),dim=1).bool(),dim=1)
                                event_mask = torch.sum(torch.sum(chunk, dim=1), dim=1, keepdim=True).bool()
                                flow_vis *= event_mask

                                vis.update(chunk_vis, label, mask, flow_vis, None)

                    # backward pass
                    if config["metrics"]["mask_events"]:
                        # event_mask = torch.unsqueeze(torch.sum(torch.sum(chunk, dim=1),dim=1).bool(),dim=1)
                        event_mask = torch.sum(torch.sum(chunk, dim=1), dim=1, keepdim=True).bool()
                        total_loss = loss_function([pred], label, mask * event_mask)
                    else:
                        total_loss = loss_function([pred], label, mask)

                    epoch_loss_valid += total_loss.item() * config["loader"]["batch_size"]
                    sample += 1
                    if sample > config['test']['sample'] // config["loader"]["batch_size"]:
                        break

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
        # default="configs/train_DSEC_supervised_MS_Spikingformer4.yml",
        default="configs/train_DSEC_supervised_MS_Spikingformer4.yml",
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



