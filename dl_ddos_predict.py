import tensorflow as tf
import numpy as np
import random as rn
import os
import csv
import pprint
from util_functions import *
# Seed Random Numbers
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
rn.seed(SEED)
config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)

from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Conv2D,Bidirectional,LSTM,ConvLSTM1D,GRU
from tensorflow.keras.layers import Dropout, GlobalMaxPooling2D,GlobalMaxPooling1D,TimeDistributed
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from lucid_dataset_parser import *
from keras_self_attention import SeqSelfAttention

import tensorflow.keras.backend as K
import pandas as pd
from multiprocessing import Queue
import time
tf.random.set_seed(SEED)
K.set_image_data_format('channels_last')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)

OUTPUT_FOLDER = "./output/"

VAL_HEADER = ['Model', 'Samples', 'Accuracy', 'F1Score', 'Hyper-parameters','Validation Set']
PREDICT_HEADER = ['Model', 'Time', 'Packets', 'Samples', 'DDOS%', 'Accuracy', 'F1Score', 'TPR', 'FPR','TNR', 'FNR', 'Source']
DDOS_HEADER={"SourceIP","SourcePort","DestIP","DestPort","Proto","Highest_layer"}
# hyperparameters
PATIENCE = 10
DEFAULT_EPOCHS = 100
#df=pd.DataFrame(columns=["SourceIP","SourcePort","DestIP","DestPort","Proto","highest_layer"])
def report_results(Y_true, Y_pred, packets, model_name, data_source, prediction_time, writer,sd_writer,keys,highest_layer):
    ddos_rate = '{:04.3f}'.format(sum(Y_pred) / Y_pred.shape[0])

    if Y_true is not None and len(Y_true.shape) > 0:  # if we have the labels, we can compute the classification accuracy
        Y_true = Y_true.reshape((Y_true.shape[0], 1))
        accuracy = accuracy_score(Y_true, Y_pred)

        f1 = f1_score(Y_true, Y_pred)
        tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred, labels=[0, 1]).ravel()
        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        tpr = tp / (tp + fn)

        row = {'Model': model_name, 'Time': '{:04.3f}'.format(prediction_time), 'Packets': packets,
               'Samples': Y_pred.shape[0], 'DDOS%': ddos_rate, 'Accuracy': '{:05.4f}'.format(accuracy), 'F1Score': '{:05.4f}'.format(f1),
               'TPR': '{:05.4f}'.format(tpr), 'FPR': '{:05.4f}'.format(fpr), 'TNR': '{:05.4f}'.format(tnr), 'FNR': '{:05.4f}'.format(fnr), 'Source': data_source}
    else:
        row = {'Model': model_name, 'Time': '{:04.3f}'.format(prediction_time), 'Packets': packets,
               'Samples': Y_pred.shape[0], 'DDOS%': ddos_rate, 'Accuracy': "N/A", 'F1Score': "N/A",
               'TPR': "N/A", 'FPR': "N/A", 'TNR': "N/A", 'FNR': "N/A", 'Source': data_source}
    pprint.pprint(row, sort_dicts=False)
    writer.writerow(row)

    #print("suspected DDOS packets details")
    index=0
    for item in Y_pred:
        if(item == 1):
            new_row={"SourceIP":str(keys[index][0]),"SourcePort":str(keys[index][1]),"DestIP":str(keys[index][2]),"DestPort":str(keys[index][3]),"Proto":str(keys[index][4]),"Highest_layer":highest_layer[index]}
            #pprint.pprint(new_row, sort_dicts=False)
            #sd_writer.writerow(zip(*(new_row[h] for h in DDOS_HEADER)))
            sd_writer.writerow(new_row)
        index=index+1
    #index=0
    #df=pd.DataFrame(columns=["SourceIP","SourcePort","DestIP","DestPort","Proto","highest_layer"])
    #for item in Y_pred:
    #    if(item == 1):
    #        new_row=pd.DataFrame([{"SourceIP":keys[index][0],"SourcePort":keys[index][1],"DestIP":keys[index][2],"DestPort":keys[index][3],"Proto":keys[index][4],"highest_layer":highest_layer[index]}])
    #        df=pd.concat([new_row,df.loc[:]]).reset_index(drop=True)
    #    index=index+1
    
    #print(df)

def start_live_capture(cap,queue):
    if isinstance(cap, pyshark.LiveCapture) == True:
        for pkt in cap.sniff_continuously():
            queue.put(pkt)
    elif isinstance(cap, pyshark.FileCapture) == True:
        while (True):
            try:
               pkt = cap.next()
               queue.put(pkt)
            except:
               print("No packets read")
               pass

def process_pcap_from_queue(queue, in_labels, max_flow_len, traffic_type='all',time_window=TIME_WINDOW):
    start_time = time.time()
    temp_dict = OrderedDict()
    labelled_flows = []

    start_time_window = start_time
    time_window = start_time_window + time_window

    while time.time() < time_window:
        try:
           pkt = queue.get(timeout=0.5)
           #if pkt is None:
           #    break
           pf = parse_packet(pkt)
           temp_dict = store_packet(pf, temp_dict, start_time_window, max_flow_len)
        except:
           break
    apply_labels(temp_dict,labelled_flows, in_labels,traffic_type)
    return labelled_flows

def main(argv):
    help_string = '''Usage: python3 dl_ddos_predict.py --predict <dataset_folder> --model <model.h5>
                            python3 dl_ddos_predict.py --predict_live <interfacename> --model <model.h5> --dataset_type <DATASET TYPE> 
                            python3 dl_ddos_predict.py --predict_live <pathofpcaporpcapng> --model <model.h5> --dataset_type <DATASET TYPE> ''' 

    parser = argparse.ArgumentParser(
        description='DDoS attacks detection with convolutional neural networks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-a', '--attack_net', default=None, type=str,
                        help='Subnet of the attacker (used to compute the detection accuracy)')

    parser.add_argument('-v', '--victim_net', default=None, type=str,
                        help='Subnet of the victim (used to compute the detection accuracy)')

    parser.add_argument('-p', '--predict', nargs='?', type=str,
                        help='Perform a prediction on pre-preprocessed data')

    parser.add_argument('-pl', '--predict_live', nargs='?', type=str,
                        help='Perform a prediction on live traffic')

    parser.add_argument('-i', '--iterations', default=1, type=int,
                        help='Predict iterations')

    parser.add_argument('-m', '--model', type=str,
                        help='File containing the model')

    parser.add_argument('-y', '--dataset_type', default=None, type=str,
                        help='Type of the dataset. Available options are: DOS2017, DOS2018, DOS2019, SYN2020')

    args = parser.parse_args()

    if os.path.isdir(OUTPUT_FOLDER) == False:
        os.mkdir(OUTPUT_FOLDER)

    if args.predict is not None:
        predict_file = open(OUTPUT_FOLDER + 'predictions-' + time.strftime("%Y%m%d-%H%M%S") + '.csv', 'a', newline='')
        predict_file.truncate(0)  # clean the file content (as we open the file in append mode)
        predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
        predict_writer.writeheader()
        predict_file.flush()

        iterations = args.iterations

        dataset_filelist = glob.glob(args.predict + "/*test.hdf5")

        if args.model is not None:
            model_list = [args.model]
        else:
            model_list = glob.glob(args.predict + "/*.h5")

        for model_path in model_list:
            model_filename = model_path.split('/')[-1].strip()
            filename_prefix = model_filename.split('-')[0].strip() + '-' + model_filename.split('-')[1].strip() + '-'
            model_name_string = model_filename.split(filename_prefix)[1].strip().split('.')[0].strip()
            K.clear_session()
            if("_ATTN" in model_path):
                model = load_model(model_path,custom_objects={"SeqSelfAttention": SeqSelfAttention})
            else:
                model = load_model(model_path)

            # warming up the model (necessary for the GPU)
            warm_up_file = dataset_filelist[0]
            filename = warm_up_file.split('/')[-1].strip()
            if filename_prefix in filename:
                X, Y = load_dataset(warm_up_file)
                if("_CONCAT" in model_path):
                     Y_pred = np.squeeze(model.predict([X,X], batch_size=1024) > 0.5)
                else:
                     Y_pred = np.squeeze(model.predict(X, batch_size=1024) > 0.5)

            for dataset_file in dataset_filelist:
                filename = dataset_file.split('/')[-1].strip()
                if filename_prefix in filename:
                    X, Y = load_dataset(dataset_file)
                    [packets] = count_packets_in_dataset([X])

                    Y_pred = None
                    Y_true = Y
                    avg_time = 0
                    for iteration in range(iterations):
                        pt0 = time.time()
                        if("_CONCAT" in model_path):
                             Y_pred = np.squeeze(model.predict([X,X], batch_size=1024) > 0.5)
                        else:
                             Y_pred = np.squeeze(model.predict(X, batch_size=1024) > 0.5)
                        pt1 = time.time()
                        avg_time += pt1 - pt0

                    avg_time = avg_time / iterations

                    report_results(np.squeeze(Y_true), Y_pred, packets, model_name_string, filename, avg_time,predict_writer,none,none)
                    predict_file.flush()

        predict_file.close()
        producer_process.join()

    if args.predict_live is not None:
        predict_file = open(OUTPUT_FOLDER + 'predictions-' + time.strftime("%Y%m%d-%H%M%S") + '.csv', 'a', newline='')
        predict_file.truncate(0)  # clean the file content (as we open the file in append mode)
        predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
        predict_writer.writeheader()
        predict_file.flush()
        sd_file = open(OUTPUT_FOLDER + 'suspectedDdos-' + time.strftime("%Y%m%d-%H%M%S") + '.csv', 'a', newline='')
        sd_file.truncate(0)  # clean the file content (as we open the file in append mode)
        sd_writer = csv.DictWriter(sd_file, fieldnames=DDOS_HEADER)
        sd_writer.writeheader()
        sd_file.flush()

        if args.predict_live is None:
            print("Please specify a valid network interface or pcap file!")
            exit(-1)
        elif ((args.predict_live.endswith('.pcap')) | (args.predict_live.endswith('.pcapng'))):
            pcap_file = args.predict_live
            cap = pyshark.FileCapture(pcap_file)
            queue = Queue()
            data_source = pcap_file.split('/')[-1].strip()
        else:
            cap =  pyshark.LiveCapture()
            queue = Queue()
            interfaces = str(args.predict_live).split(',')
            cap.interfaces = interfaces
            data_source = args.predict_live
        capture_process = Process(target=start_live_capture, args=(cap,queue,))
        capture_process.daemon = True
        capture_process.start()
        print ("Prediction on network traffic from: ", data_source)

        
        # load the labels, if available
        labels = parse_labels(args.dataset_type, args.attack_net, args.victim_net)

        # do not forget command sudo ./jetson_clocks.sh on the TX2 board before testing
        if args.model is not None and args.model.endswith('.h5'):
            model_path = args.model
        else:
            print ("No valid model specified!")
            exit(-1)

        model_filename = model_path.split('/')[-1].strip()
        filename_prefix = model_filename.split('n')[0] + 'n-'
        time_window = int(filename_prefix.split('t-')[0])
        max_flow_len = int(filename_prefix.split('t-')[1].split('n-')[0])
        model_name_string = model_filename.split(filename_prefix)[1].strip().split('.')[0].strip()
        K.clear_session()
        if("_ATTN" in model_path):
             model = load_model(model_path,custom_objects={"SeqSelfAttention": SeqSelfAttention})
        else:
             model = load_model(args.model)

        mins, maxs = static_min_max(time_window)
        tolerance=0
        while (True):
            samples = process_pcap_from_queue(queue, labels, max_flow_len, traffic_type="all", time_window=time_window)
            if len(samples) > 0:
                X,Y_true,keys,highest_layer = dataset_to_list_of_fragments(samples)
                X = np.array(normalize_and_padding(X, mins, maxs, max_flow_len))
                if labels is not None:
                    Y_true = np.array(Y_true)
                else:
                    Y_true = None

                X = np.expand_dims(X, axis=3)
                pt0 = time.time()
                if("_CONCAT" in model_path):
                    Y_pred = np.squeeze(model.predict([X,X], batch_size=1024) > 0.5,axis=1)
                else:
                    Y_pred = np.squeeze(model.predict(X, batch_size=1024) > 0.5,axis=1)
                pt1 = time.time()
                prediction_time = pt1 - pt0

                [packets] = count_packets_in_dataset([X])
                report_results(np.squeeze(Y_true), Y_pred, packets, model_name_string, data_source, prediction_time,predict_writer,sd_writer,keys,highest_layer)
                predict_file.flush()
                sd_file.flush()
                tolerance= 0
            elif isinstance(cap, pyshark.LiveCapture) == True:
                if(tolerance == 0):
                    time.sleep(0.5)
                    tolerance= tolerance + 1
                    continue
                print("No packets available")
                capture_process.terminate()
                time.sleep(0.1)
                break

            elif isinstance(cap, pyshark.FileCapture) == True:
                print("\nNo more packets in file ", data_source)
                capture_process.terminate()
                time.sleep(0.1)
                break
          

        predict_file.close()
        sd_file.close()
        capture_process.join() 
        #start_processing.join()
                   


if __name__ == "__main__":
    main(sys.argv[1:])
