# Cyber Security: Near Real Time DDoS Cyber Attack Detection Using Deep Learning Method 

DL_DDOS Models are a group of Deep Learning Models built for detecting the DDOS attack . This is based on  LUCID(Lightweight, Usable CNN in DDoS Detection) and utilizes the traffic and dataset parser. This iproject reuses the LUCID's dataset-agnostic pre-processing mechanism that produces traffic observations consistent with those collected in existing online systems, where the detection algorithms must cope with segments of traffic flows collected over pre-defined time windows.


## Installation

The DL_DDOS Models are implemented in Python v3.9 with Keras and Tensorflow 2, while the traffic pre-processing tool is implemented in Python v3.9, Numpy and Pyshark.
This project is validated in Ubuntu 20.04.5 LTS (x86_64 GNU/Linux)
Following steps are needed for bringing up the DL_DDOS Models

sudo apt-get install python3.9-dev python3.9-venv

python3.9 -m venv ~/ddos/py39 

source ~/ddos/py39/bin/activate

pip install -U 'protobuf==3.9.2'

pip install tensorflow==2.7.0

pip install scikit-learn

pip install keras_self_attention

#pip install pyshark sklearn numpy tensorflow==2.7.0 h5py lxml
```

Pyshark is just Python wrapper for tshark, allowing python packet parsing using wireshark dissectors. This means that ```tshark``` must be also installed. On an Ubuntu-based OS, use the following command:

```
sudo apt install tshark
```

Please note that the current  code works with ```tshark``` **version 3.2.13 or lower**. Issues have been reported when using newer releases such as 3.4.X.

For the sake of simplicity, we omit the command prompt ```(python39)$``` in the following example commands in this README.   ```(python39)$``` indicates that we are working inside the ```python39``` execution environment, which provides all the required libraries and tools. If the command prompt is not visible, re-activate the environment as explained above.

## Traffic pre-processing

LUCID requires a labelled dataset, including the traffic traces in the format of ```pcap``` files. The traffic pre-processing functions are implemented in the ```lucid_dataset_parser.py``` Python script. It currently supports three DDoS datasets from the University of New Brunswick (UNB) (https://www.unb.ca/cic/datasets/index.html): CIC-IDS2017, CSE-CIC-IDS2018 and CIC-DDoS2019, plus a custom dataset containing a SYN Flood DDoS attack (SYN2020) that will be used for this guide and included in the ```sample-dataset``` folder.

With term *support*, we mean the capability of the script to correctly label the packets and the traffic flows either as benign or DDoS. In general, this is done by parsing a file with the labels provided with the traffic traces, like in the case of the UNB datasets, or by manually indicating the IP address(es) of the attacker(s) and the IP address(es) of the victim(s) in the code. Of course, also in the latter case, the script must be tuned with the correct information of the traffic (all the attacker/victim pairs of IP addresses), as this information is very specific to the dataset and to the methodology used to generate the traffic. 

Said that, ```lucid_dataset_parser.py``` includes the structures with the pairs attacker/victim of the three datasets mentioned above (CIC-IDS2017, CSE-CIC-IDS2018, CIC-DDoS2019 and SYN2020), but it can be easily extended to support other datasets by replicating the available code.

For instance, the following Python dictionary provides the IP addresses of the 254 attackers and the victim involved in the custom SYN Flood attack:   

```
CUSTOM_DDOS_SYN = {'attackers': ['11.0.0.' + str(x) for x in range(1,255)],
                      'victims': ['10.42.0.2']}
```

### Command options

The following parameters can be specified when using ```lucid_dataset_parser.py```:

- ```-d```, ```--dataset_folder```: Folder with the dataset
- ```-o```, ```--output_folder ```: Folder where  the scripts saves the output. The dataset folder is used when this option is not used
- ```-f```, ```--traffic_type ```: Type of flow to process (all, benign, ddos)
- ```-p```, ```--preprocess_folder ```: Folder containing the intermediate files ```*.data```
- ```-t```, ```--dataset_type ```: Type of the dataset. Available options are: DOS2017, DOS2018, DOS2019, SYN2020
- ```-n```, ```--packets_per_flow ```: Maximum number of packets in a sample
- ```-w```, ```--time_window ```: Length of the time window (in seconds)
- ```-i```, ```--dataset_id ```: String to append to the names of output files



### First step

The traffic pre-processing operation comprises two steps. The first parses the file with the labels (if needed) all extracts the features from the packets of all the ```pcap``` files contained in the source directory. The features are grouped in flows, where a flow is a set of features from packets with the same source IP, source UDP/TCP port, destination IP and destination UDP/TCP port and protocol. Flows are bi-directional, therefore, packet (srcIP,srcPort,dstIP,dstPort,proto) belongs to the same flow of (dstIP,dstPort,srcIP,srcPort,proto). The result is a set of intermediate binary files with extension ```.data```.

This first step can be executed with command:

```
python3 lucid_dataset_parser.py --dataset_type SYN2020 --dataset_folder ./sample-dataset/ --packets_per_flow 10 --dataset_id SYN2020 --traffic_type all --time_window 10
```

This will process in parallel the two files, producing a file named ```10t-10n-SYN2020-preprocess.data```. In general, the script loads all the ```pcap``` files contained in the folder indicated with option ```--dataset_folder``` and starting with prefix ```dataset-chunk-```. The files are processed in parallel to minimise the execution time.

Prefix ```10t-10n``` means that the pre-processing has been done using a time window of 10 seconds (10t) and a flow length of 10 packets (10n). Please note that ```SYN2020``` in the filename is the result of option ```--dataset_id SYN2020``` in the command.

Time window and flow length are two hyperparameters of LUCID. For more information, please refer to the research paper mentioned above. 

### Second step

The second step loads the ```*.data``` files, merges them into a single data structure stored in RAM memory,  balances the dataset so that number of benign and DDoS samples are approximately the same, splits the data structure into training, validation and test sets, normalises the features between 0 and 1 and executes the padding of samples with zeros so that they all have the same shape (since having samples of fixed shape is a requirement for a CNN to be able to learn over a full sample set).

Finally, three files (training, validation and test sets) are saved in *hierarchical data format* ```hdf5``` . 

The second step is executed with command:

```
python3 lucid_dataset_parser.py --preprocess_folder ./sample-dataset/
```

If option ```--output_folder``` is not used, the output will be produced in the input folder specified with option ```--preprocess_folder```.

At the end of this operation, the script prints a summary of the pre-processed dataset. In our case, with this tiny traffic traces, the result should be something like:

```
2020-08-27 11:02:20 | examples (tot,ben,ddos):(3518,1759,1759) | Train/Val/Test sizes: (2849,317,352) | Packets (train,val,test):(15325,1677,1761) | options:--preprocess_folder ./sample-dataset/ |
```

Which means 3518 samples in total (1759 benign and 1759 DDoS), 2849 in the training set, 317 in the validation set and 352 in the test set. The output also shows the total number of packets in the dataset divided in training, validation and test sets and the options used with the script. 

All the output of the ```lucid_dataset_parser.py``` script is saved within the output folder in the ```history.log``` file.

## Training

The DL_DDOS Models with Two Dense Layers are implemented in dl_ddos_models.py.

The DL_DDOS Models with Global Max Pooling Layer replacing the flattening and once Dense Layer is implemented in dl_ddos_models_max_pool.py

All the Models are configured with default Learning rate (0.01) , L2 Regularization and Batch size of 1024

The training continues until the maximum number of epochs is reached or after the loss has not decreased for 10 consecutive times. This value is defined with variable ```PATIENCE=10``` at the beginning of the script. Part of the hyperparameters is defined in the script as follows:


Other two important hyperparameters must be specified during the first step of the data preprocessing (see above):

- **Maximum number of packets/sample (n)**: indicates the maximum number of packets of a flow recorded in chronological order in a sample.
- **Time window (t)**: Time window (in seconds) used to simulate the capturing process of online systems by splitting the flows into subflows of fixed duration.

To tune Models with this two hyperparameters, the data preprocessing step must be executed multiple times to produce different versions of the dataset, one for each combination of **n** and **t**. This of course will produce multiple versions of the dataset with different prefixes like: ```10t-10n```, ```10t-100n```, ```100t-10n``` and ```100t-100n``` when testing with ```n=10,100``` and ```t=10,100```.

All these files can be stored into a single folder, or in multiple  subfolders. The script takes care of loading all the versions of the dataset available in the folder (and its subfolders) specified with option ```--dataset_folder```, as described below.

All the evaluation is done using 10t-10n combination.

### Command options

To execute the training process, the following parameters can be specified when using ```dl_ddos_models.py```:
To train model using Global Max pool , use dl_ddos_models_max_pool.py

- ```-t```, ```--train```: Starts the training process and specifies the folder with the dataset
- ```-e```, ```--epochs ```: Maximum number of training epochs for each set of hyperparameters (default=1000)
     -mn  ,    --modelname : Build the specified model .Default is LSTM and other supported values are BI_LSTM, LSTM_ATTN, BI_LSTM_ATTN, GRU, BI_GRU, BI_GRU_ATTN,CONVLSTM1D,BI_LSTM_ATTN_GRU

### The training process

To train DL_DDOS, execute the following command:
```
python3 dl_ddos_models.py --train ./sample-dataset  --modelname BI_LSTM_ATTN
#python3 dl_ddos_models.py --train ./<datasetfolder>  --modelname BI_LSTM_ATTN
```

This command trains the models for maximum 100 epochs . The training process can stop earlier if no progress towards the minimum loss is observed for PATIENCE=10 consecutive epochs. The model which maximises the accuracy on the validation set is saved in ```h5``` format in the ```output``` folder, along with a ```csv``` file with the performance of the model on the validation set.  The name of the two files is the same (except for the extension) and is in the following format:

```
10t-10n-IDS201X-BI_LSTM_ATTN.h5   : IDS201X is mentioned here inplace of SYN2020 as the data set used is a combination  dataset. 
10t-10n-IDS201X-BI_LSTM_ATTN.csv
Folder with modelname is created which contain the saved model,weights
10t-10n-SYN2020-BI_LSTM_ATTN
```

Where the prefix 10t-10n indicates the values of hyperparameters ```time window``` and ```packets/sample``` that produced the best results in terms of F1 score on the validation set. The values of the other hyperparameters are reported in the ```csv``` file:

Model	                Samples	Accuracy  F1Score	Hyper-parameters	Validation Set
IDS201X-BILSTMi-ATTN	247334	0.9892	   0.9893       ""	   	./final_dataset/10t-10n-IDS201X-dataset-val.hdf5

## Testing

Testing means evaluating a trained model of LUCID with unseen data (data not used during the training and validation steps), such as the test set in the ```sample-dataset``` folder. For this process,  the ```lucid_cnn.py``` provides a different set of options:

- ```-p```, ```--predict```: Perform prediction on the test sets contained in a given folder specified with this option. The folder must contain files in ```hdf5``` format with the ```test``` suffix
- ```-m```, ```--model```: Model to be used for the prediction. The model in ```h5``` format produced with the training
- ```-i```, ```--iterations```: Repetitions of the prediction process (useful to estimate the average prediction time)

To test DL MODELS , run the following command:

```
python3 dl_ddos_models.py --predict ./<datasetfoldernamewhere test data is present>/ --model ./output/10t-10n-IDS201X-BI_LSTM_ATTN.h5
python3 dl_ddos_models_max_pool.py --predict ./<datasetfoldernamewhere test data is present>/ --model ./output/10t-10n-IDS201X-GM-BI_LSTM_ATTN.h5
```

The output printed on the terminal and saved in a text file in the ```output``` folder in the following format:

|Model|Time|Packets|Samples|DDOS%|Accuracy|F1Score|TPR|FPR|TNR|FNR|Source|
|-----|----|-------|-------|-----|--------|-------|---|---|---|---|------|
IDS201X-BI_LSTM_ATTN,4.469,980437,274813,0.505,0.9938,0.9938,0.9964,0.0088,0.9912,0.0036,10t-10n-IDS201X-dataset-test.hdf5

Where ```Time``` is the execution time on a test set.  The values of ```Packets``` and ```Samples``` are the the total number of packets and samples in the test set respectively. More precisely, ```Packets``` is the total amount of packets represented in the samples (traffic flows) of the test set. ```Accuracy```, ```F1```, ```PPV```  are classification accuracy, F1 and precision scores respectively, ```TPR```, ```FPR```, ```TNR```, ```FNR``` are the true positive, false positive, true negative and false negative rates respectively. 

The last column indicates the name of the test set used for the prediction test. Note that the script loads and process all the test sets in the folder specified with option ``` --predict``` (identified with the suffix ```test.hdf5```). This means that the output might consist of multiple lines, on for each test set. 

## Online Inference

Once trained, DL_DDOS Models can perform inference on live network traffic or on pre-recorded traffic traces saved in ```pcap``` format. This operational mode is implemented in the ```dl_ddos_models.py,`dl_ddos_models_max_pool.py``` script and leverages on ```pyshark``` and ```tshark``` tools to capture the network packets from one of the network cards of the machine where the script is executed, or to extract the packets from a ```pcap``` file. In both cases, the script simulates an online deployment, where the traffic is collected for a predefined amount of time (```time_window```) and then sent to the neural network for classification.

Online inference can be started by executing ```dl_ddos_models.py``` followed by one or more of these options: 

- ```-pl```, ```--predict_live```: Perform prediction on the network traffic sniffed from a network card or from a ```pcap``` file available on the file system. Therefore, this option must be followed by either the name of a network interface (e.g., ```eth0```) or the path to a ```pcap``` file (e.g., ```/home/user/traffic_capture.pcap```)
- ```-m```, ```--model```: Model to be used for the prediction. The model in ```h5``` format produced with the training
- ```-y```, ```--dataset_type```: One between ```DOS2017```, ```DOS2018``` and ```DOS2019``` in the case of ```pcap``` files from the UNB's datasets, or ```SYN2020``` for the custom dataset provided with this code. This option is not used by LUCID for the classification task, but only to produce the classification statistics (e.g., accuracy, F1 score, etc,) by comparing the ground truth labels with the LUCID's output
- ```-a```, ```--attack_net```: Specifies the subnet of the attack network (e.g., ```192.168.0.0/24```). Like option ```dataset_type```, this is used to generate the ground truth labels. This option is used, along with option ```victim_net```, in the case of custom traffic or pcap file with IP address schemes different from those in the three datasets ```DOS2017```,  ```DOS2018```, ```DOS2019```  or ```SYN2020``` 
- ```-y```, ```--victim_net```: The subnet of the victim network (e.g., ```10.42.0.0/24```), specified along with option ```attack_net``` (see description above).

### Inference on live traffic

If the argument of ```predict_live``` option is a network interface, DL_MODELS will sniff the network traffic from that interface and will return the classification results every time the time window expires. The duration of the time window is automatically detected from the prefix of the model's name (e.g., ```10t``` indicates a 10-second time window). To start the inference on live traffic, use the following command:

```
python3 dl_ddos_models.py --predict_live eth0 --model ./output/10t-10n-IDS201X-BI_LSTM_ATTN.h5 --dataset_type SYN2020 
```

Where ```eth0``` is the name of the network interface, while ```dataset_type``` indicates the address scheme of the traffic. This is optional and, as written above, it is only used to obtain the ground truth labels needed to compute the classification accuracy.

In the example, ```SYN2020``` refers to a SYN flood attack built using the following addressing scheme, defined in ```lucid_dataset_parser.py```, and used in the sample dataset:

```
CUSTOM_DDOS_SYN = {'attackers': ['11.0.0.' + str(x) for x in range(1,255)],
                      'victims': ['10.42.0.2']}
```

Of course, the above dictionary can be changed to meet the address scheme of the network where the experiments are executed. Alternatively, one can use the ```attack_net``` and ```victim_net``` options as follows:

```
python3 dl_ddos_models.py --predict_live eth0 --model ./output/10t-10n-IDS201X-BI_LSTM_ATTN.h5 --attack_net 11.0.0.0/24 --victim_net 10.42.0.0/24
```

Once DL Models has been started on the victim machine using one of the two examples above, we can start the attack from another host machine using one of the following scripts based on the ```mausezahn``` tool (https://github.com/uweber/mausezahn):

```
sudo mz eth0 -A  11.0.0.0/24 -B 10.42.0.2 -t tcp " dp=80,sp=1024-60000,flags=syn"
```

In this script, ```eth0``` refers to the egress network interface on the attacker's machine and ```10.42.0.2``` is the IP address of the victim machine.

The output of DL MODELS on the victim machine will be similar to that reported in Section **Testing** above. 

### Inference on pcap files

Similar to the previous case on live traffic, inference on a pre-recorded traffic trace can be started with command:

```
python3 dl_ddos_models.py --predict_live ./sample-dataset/dataset-chunk-syn.pcap --model ./output/10t-10n-IDS201X-BI_LSTM_ATTN.h5 --dataset_type SYN2020
```

In this case, the argument of option ```predict_live``` must be the path to a pcap file. The script parses the file from the beginning to the end, printing the classification results every time the time window expires. The duration of the time window is automatically detected from the prefix of the model's name (e.g., ```10t``` indicates a 10-second time window). 

The output of DL Models on the victim machine will be similar to that reported in Section **Testing** above. 

### Incremental Learning 

For Incremental Learning following need to be done:
Create <newdata>.data file as in step 1 .
Copy this file to the folder where <current>.data files are located.
Run step2 on this folder. The output of the preprocesser will be combination of old and new dataset.

During training specify "--incremental True " along with step 3
python3 dl_ddos_models.py --train ./<datasetfolder>  --modelname BI_LSTM_ATTN --incremental True

This will take the saved model from the folder created during earlier training and load the model and the weights and losses. The Training will be faster and will get stopped with less number of epochs compared to full training.

