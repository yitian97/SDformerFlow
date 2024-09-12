# SDformer-Flow: Spiking Neural Network Transformer for Event-based optical flow estimation


This code allows for the reproduction of our paper:

SDformerFlow: Spiking Neural Network Transformer for Event-based Optical Flow, ICPR2024

and our new improved version of the model:

SDformerFlow: Spatiotemporal swin spikeformer for event-based optical flow estimation. [[arxiv]](https://arxiv.org/abs/2409.04082)

The following are results tested on our validation dataset on DSEC dataset. Flow estimation are masked where valid ground truth data is available.

<!-- &nbsp; -->
<img src="SDFormerflow.gif" width="864" height="168" />
<!-- &nbsp; -->

The following are dense flow results tested on official DSEC test dataset:

<!-- &nbsp; -->
<img src="DSEC-test.gif" width="800" height="224" />
<!-- &nbsp; -->

## Environment
The code is tested on Ubuntu 22.04 with Cuda 11.8 and Python 3.7.3. 
It is recommended to use conda enviornment:
```
conda create -n SDformerflow python=3.7.3
conda activate SDformerflow
```
Install the dependencies:
```
pip install -r requirements.txt
```


In `configs/`, you can find the configuration files associated to these scripts and vary the inference settings (e.g., number of input events, learning rate).

We use [MLflow](https://www.mlflow.org/docs/latest/index.html#) to log the training process. 


## Dataset preprocessing
DSEC dataset can be downloaded in the [DSEC dataset](https://dsec.ifi.uzh.ch/dsec-datasets/download/).

MVSEC dataset can be downloaded in the [MVSEC dataset](https://daniilidis-group.github.io/mvsec/).

MDR dataset can be downloaded in the [MDR dataset](https://daniilidis-group.github.io/mvsec/).

For DSEC dataset, the data is preprocessed using `DSEC_dataloader/DSEC_dataset_preprocess.py` script.
We follow the same data splits as in [OF_EV_SNN](https://github.com/J-Cuadrado/OF_EV_SNN).
The preprocessed data is stored in the `data/` folder.

All the data should be stored in the following structure:
```
data/
    ├── Dataset/
    │      ├─── DSEC/
    │      │    └─── saved_flow_data/
    │      │          ├── event_tensors/
    │      │          │   ├── 10bins/
    │      │          │   |   ├── left/
    │      │          │   |   └── ...
    |      │          |   └── ...
    │      │          ├── gt_tensors/
    │      │          ├── mask_tensors/
    │      │          └── sequence_lists/
    │      ├─── MDR/
    │      │    ├─── dt1/
    │      │    │     ├── train/
    │      │    │     └── test/
    │      │    └───  dt4/
    │      └─── MVSEC/
    └── ...  
```
## Inference

To estimate optical flow from event sequences from the DSEC dataset and compute the AEE, AAE and percentage of outliers, run:

```
python eval_DSEC_flow_SNN.py --config configs/valid_DSEC_Supervised.yml
```

To estimate optical flow from event sequences from the MVSEC dataset and compute the AEE, AAE and percentage of outliers, run:

```
python eval_MV_flow_SNN.py --config configs/eval_MV_supervised.yml
```
## Training
For training STTFlowNet on the DSEC dataset, run:
```
python train_flow_parallel_supervised.py --config configs/train_DSEC_supervised_STT_voxel.yml
```
For training SDformerFlow on the DSEC dataset, run:

```
python train_flow_parallel_supervised_SNN.py --config configs/train_DSEC_supervised_MS_Spikingformer4.yml
```
For training STTFlowNet on the MDR dataset, run:
```
python train_mdr_supervised_ANN.py --config configs/train_MDR_supervised_STT_voxel.yml
```
For training SDformerFlow on the MDR dataset, run:

```
python train_mdr_supervised_SNN.py --config configs/train_MDR_supervised_MS_Spikingformer.yml
```

## Please cite our paper if you find this code useful:
```
@inproceedings{tian2024sdformerflow,
               title={SDformerFlow: Spiking Neural Network Transformer for Event-based Optical Flow},
               author={Tian, Yi and Andrade-Cetto, Juan},
               booktitle={International Conference on Pattern Recognition (ICPR)},
               year={2024},
}
```
```
@misc{tian2024sdformerflow2,
      title={SDformerFlow: Spatiotemporal swin spikeformer for event-based optical flow estimation}, 
      author={Yi Tian and Juan Andrade-Cetto},
      year={2024},
      eprint={2409.04082},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.04082}, 
}
```
## Thanks to the following open-source projects:


The spatial-temporal swin spikeformer module is adapted from the following project:
[Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer)


Spiking Neurals Networks are implemented using Spikingjelly library:
[SpikingJelly](https://github.com/fangwei123456/spikingjelly)


For event data preprocessing, refer to the following project:
[E-RAFT: Dense Optical Flow from Event Cameras](https://github.com/uzh-rpg/E-RAFT); 
[Optical Flow estimation from Event Cameras and Spiking Neural Networks](https://github.com/j-cuadrado/of_ev_snn)



