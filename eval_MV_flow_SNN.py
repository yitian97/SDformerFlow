import torch
import argparse
import mlflow
from configs.parser import YAMLParser
from loss.flow_supervised import *
from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
from tqdm import tqdm
from utils.mlflow import log_config, log_results
from utils.utils import load_model,  create_model_dir,save_csv, save_model, count_parameters,print_parameters
from utils.visualization import Visualization_DSEC
from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite
from DSEC_dataloader.data_augmentation import downsample_data,Compose,CenterCrop,RandomCrop,RandomRotationFlip,Random_event_drop
import math
from spikingjelly.activation_based import functional,neuron
from models.STSwinNet_SNN.Spiking_submodules import *
import random


use_ml_flow = True



def valid_test(args, config_parser):
    ########## configs ##########
    config = config_parser.config
    path = 'output/' + str(args.runid) + '/'
    sequence = 'valid_remap'
    if use_ml_flow:
        mlflow.set_tracking_uri(args.path_mlflow)
        run = mlflow.get_run(args.runid)
        config = config_parser.merge_configs(run.data.params)
        # create directory for inference results
        path_results = create_model_dir(args.path_results, args.runid)


        # store validation settings
        eval_id = log_config(path_results, args.runid, config)



    # initialize settings
    device = config_parser.device
    loss_function = flow_loss_supervised(config, device)
    # visualization tool
    if config["vis"]["enabled"] or config["vis"]["store"] or config["vis"]["store_att"]:
        vis = Visualization_DSEC(config, eval_id=eval_id, path_results=path_results)

    ########## data loader ##########
    if config["loader"]["crop"] is not None:
        transform_valid = Compose([CenterCrop((config['loader']['crop'][0], config['loader']['crop'][1])) ])
        config["swin_transformer"]["input_size"] = [config['loader']['crop'][0], config['loader']['crop'][1]]
    else:
        transform_valid = None
        config["swin_transformer"]["input_size"] = [config['loader']['resolution'][0], config['loader']['resolution'][1]]


    # Create validation dataset

    file = "valid"

    print("Creating Validation Dataset ...")
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


    # Define validation dataloader
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=config["loader"]["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )





    # Transformer config
    if config["swin_transformer"]["use_arc"][0]:
        model = eval(config["model"]["name"])(config["model"].copy(), config["swin_transformer"].copy())
    else:
        model = eval(config["model"]["name"])(config["model"].copy())



    model.to(device)
    model.init_weights()

    remap = config["loader"]["remap"] if "remap" in config["loader"] else None


    model = load_model(args.runid, model, device, remap = remap, test = True) # delete the relative positioning bias and index



    #reset SNN, step model, backend
    functional.reset_net(model)
    functional.set_step_mode(model, config['data']['step_mode'])
    if device.type != 'cpu':
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
        functional.set_backend(model, "cupy", neurontype)



    print_parameters(model)
    print("Total parameters: ", count_parameters(model))


    # Define the loss function

    loss_function = flow_loss_supervised(config,device)

    # Validation Dataset
    model.eval()

    print('Validating... (test sequence)')
    sample = 0
    val_results = {}
    for metric in config["metrics"]["name"]:
        val_results[metric] = {}
        val_results[metric]["metric"] = 0
        val_results[metric]["it"] = 0
        if metric == "AEE":
            val_results[metric]["PE1"] = 0
            val_results[metric]["PE2"] = 0
            val_results[metric]["PE3"] = 0
            val_results[metric]["outliers"] = 0

    for data in tqdm(valid_dataloader):

        functional.reset_net(model)
        functional.set_step_mode(model, config['data']['step_mode'])  # layer-by-layer

        chunk = data['event_volume_new'].to(device=device, dtype=torch.float32)
        if config["data"]["num_chunks"] == 2:
            chunk_old = data['event_volume_old'].to(device=device, dtype=torch.float32)
            chunk = torch.cat((chunk_old, chunk), dim=1)
        label = data['flow'].to(device=device, dtype=torch.float32)
        mask = data['valid'].unsqueeze(dim=1).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # forward pass
            if config['model']['encoding'] == 'cnt':
                chunk_vis = torch.sum(chunk, dim=2)
                chunk = chunk.view([chunk.shape[0], -1] + list(chunk.shape[3:]))#for EV-FlowNet  #[B,40,H,W] 2+2+2...

            elif config['model']['encoding'] == 'voxel':
                # ignore polarity
                if config['loader']['polarity']:
                    neg = torch.nn.functional.relu(-chunk)
                    pos = torch.nn.functional.relu(chunk)
                    chunk = torch.cat((torch.unsqueeze(pos, dim=2), torch.unsqueeze(neg, dim=2)),
                                      dim=2)  # B,C=20,P=2,H,W   B C, P, H, W

                    if config["vis"]["enabled"] or config["vis"]["store_att"] or config["vis"]["store"]:
                        chunk_vis = torch.stack((torch.sum(pos, dim=1), torch.sum(neg, dim=1)), dim=1)

                else:
                    if config["vis"]["enabled"] or config["vis"]["store_att"] or config["vis"]["store"]:
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

        if config['metrics']['mask_events']:
            event_mask = torch.sum(torch.sum(chunk, dim=1), dim=1, keepdim=True).bool()
            mask = mask * event_mask

        total_loss = loss_function([pred], label, mask)
        print(total_loss)

        if config["vis"]["enabled"]  or config["vis"]["store_att"] or config["vis"]["store"] and config["loader"]["batch_size"] == 1:
            flow_vis = pred.clone()
            flow_vis *= mask


        # validation
        # validation metric
        criteria = []
        if "metrics" in config.keys():
            for metric in config["metrics"]["name"]:
                criteria.append(eval(metric)(pred, label, mask, config['metrics']['flow_scaling']))
        for i, metric in enumerate(config["metrics"]["name"]):
                # compute metric
                val_metric = criteria[i]()

                # # accumulate results
                for batch in range(config["loader"]["batch_size"]):
                    val_results[metric]["it"] += 1
                    if metric == "AEE":
                        val_results[metric]["metric"] += val_metric[0][batch].cpu().numpy()
                        val_results[metric]["PE1"]  += val_metric[1][batch].cpu().numpy()
                        val_results[metric]["PE2"]  += val_metric[2][batch].cpu().numpy()
                        val_results[metric]["PE3"]  += val_metric[3][batch].cpu().numpy()
                        val_results[metric]["outliers"]  += val_metric[4][batch].cpu().numpy()
                    else:
                        val_results[metric]["metric"] += val_metric[batch].cpu().numpy()



        with torch.no_grad():
            if config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                vis.update(chunk_vis, label*mask, mask, flow_vis, None)
            if config["vis"]["store"]:
                sequence = sequence
                vis.store(chunk_vis, label*mask, mask, flow_vis, sequence, None)
        sample += 1




    # store validation config and results
    results = {}
    if "metrics" in config.keys():
        for metric in config["metrics"]["name"]:
            results[metric] = {}
            if metric == "AEE":
                results[metric + "_PE1"] = {}
                results[metric + "_PE2"] = {}
                results[metric + "_PE3"] = {}
                results[metric + "_outliers"] = {}

            results[metric] = str(val_results[metric]["metric"] / val_results[metric]["it"])
            if metric == "AEE":
                results[metric + "_PE1"] = str(
                    val_results[metric]["PE1"] / val_results[metric]["it"]
                )
                results[metric + "_PE2"] = str(
                    val_results[metric]["PE2"] / val_results[metric]["it"]
                )
                results[metric + "_PE3"] = str(
                    val_results[metric]["PE3"] / val_results[metric]["it"]
                )
                results[metric + "_outliers"] = str(
                    val_results[metric]["outliers"] / val_results[metric]["it"]
                )


            log_results(args.runid, results, path_results, eval_id)

            print(results[metric], results[ "AEE_PE1"], results["AEE_PE2"], results["AEE_PE3"], results["AEE_outliers"] )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/eval_MV_supervised.yml",
        help="training configuration",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )

    parser.add_argument("--runid", default="a823f1a0ec8f4c3599affe1955e91339", help="mlflow run")  # STT trainin cropped
    parser.add_argument(
        "--save_path",
        default="results/checkpoint_epoch{}.pth",
        help="save the model",
    )
    parser.add_argument("--path_results", default="results_inference/")

    parser.add_argument("--mode", default="valid", help="valid or test")

    args = parser.parse_args()

    # launch test
    if args.mode == "valid":
        valid_test(args, YAMLParser(args.config))
