import tensorflow as tf
import numpy as np
import random as rn
import os
import csv
import pprint
from util_functions import *
from collections import deque
import time
# Seed Random Numbers
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
rn.seed(SEED)
config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)

from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Conv2D,Bidirectional,LSTM,ConvLSTM1D,GRU,concatenate
from tensorflow.keras.layers import Dropout, GlobalMaxPooling2D,GlobalMaxPooling1D,TimeDistributed,InputLayer,LayerNormalization,BatchNormalization
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from lucid_dataset_parser import *
from keras_self_attention import SeqSelfAttention
from keras_multi_head import MultiHead
import tensorflow.keras.backend as K
tf.random.set_seed(SEED)
K.set_image_data_format('channels_last')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

OUTPUT_FOLDER = "./output/"

VAL_HEADER = ['Model', 'Samples', 'Accuracy', 'F1Score', 'Hyper-parameters','Validation Set']
PREDICT_HEADER = ['Model', 'Time', 'Packets', 'Samples', 'DDOS%', 'Accuracy', 'F1Score', 'TPR', 'FPR','TNR', 'FNR', 'Source']

# hyperparameters
PATIENCE = 10
DEFAULT_EPOCHS = 100

def build_model(dataset_name,model_name,input_shape):
    K.clear_session()
    model_full_name=dataset_name+"-"+model_name
    model = Sequential(name=model_full_name)

    if (model_name == "BI_LSTM_ATTN"):
        model.add(Bidirectional(LSTM(32, activation='tanh', kernel_regularizer='l2',return_sequences='true'),input_shape=input_shape,name="BI_LSTM_ATTN"))
        model.add(LayerNormalization())
        model.add(SeqSelfAttention(attention_activation='sigmoid',name='Attention'))
        model.add(LayerNormalization())
        model.add(Flatten())
    elif(model_name== "BI_LSTM_MH_ATTN"):
        model.add(MultiHead(
        layer=Bidirectional(LSTM(units=32, activation='tanh', kernel_regularizer='l2',return_sequences='true'),name='BI_LSTM'),
        input_shape=input_shape,
        layer_num=3,
        reg_index=[1],
        #reg_index=[1, 4],
        #reg_slice=(slice(None, None), slice(32, 48)),
        #reg_slice=(slice(None, None), slice(6, 10)),
        reg_factor=0.1,
        name='Multi-Head',
        ))
        #model.add(LayerNormalization())
        #model.add(SeqSelfAttention(attention_activation='sigmoid',name='Attention'))
        model.add(Flatten(name='Flatten'))
        #model.add(BatchNormalization())

    elif (model_name == "SBL_A"):
        model.add(Bidirectional(LSTM(32, activation='tanh', kernel_regularizer='l2',return_sequences='true'),input_shape=input_shape,name="BI_LSTM_1"))
        model.add(LayerNormalization())
        model.add(Bidirectional(LSTM(32, activation='tanh', kernel_regularizer='l2',return_sequences='true'),name="BI_LSTM_2"))
        model.add(LayerNormalization())
        model.add(SeqSelfAttention(attention_activation='sigmoid',kernel_regularizer='l2',
                       bias_regularizer='l1',
                       attention_regularizer_weight=1e-4,name='Attention1'))
        model.add(LayerNormalization())
        model.add(Dropout(0.5))
        model.add(Flatten())
    elif (model_name == "SBLBG_A"):
        model.add(Bidirectional(LSTM(32, activation='tanh', kernel_regularizer='l2',return_sequences='true'),input_shape=input_shape,name="BI_LSTM_ATTN"))
        model.add(SeqSelfAttention(attention_activation='sigmoid',name='Attention1'))
        model.add(LayerNormalization())
        model.add(Bidirectional(GRU(32, activation='tanh', kernel_regularizer='l2',return_sequences='true'),name="BI_LSTM_ATTN_GRU"))
        model.add(SeqSelfAttention(attention_activation='sigmoid',name='Attention2'))
        model.add(LayerNormalization())
        model.add(Dropout(0.5))
        model.add(Flatten())
    # Final Model for BI_GRU_ATTN
    elif(model_name == "BI_GRU_ATTN"):
        model.add(Bidirectional(GRU(32, activation='tanh', kernel_regularizer='l2',return_sequences='true'),input_shape=input_shape,name="BI_GRU_ATTN"))
        model.add(LayerNormalization())
        model.add(SeqSelfAttention(kernel_regularizer='l2', attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,name='Attention'))
        model.add(LayerNormalization())
        model.add(Flatten())
    else:
        print("Unknown Model")
    model.add(Dense(32, activation = 'relu', kernel_regularizer='l2'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation = 'sigmoid', kernel_regularizer='l2'))
    print(model.summary())
    #compileModel(model,0.0001)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
def compileModel(model,lr):
    # optimizer = SGD(learning_rate=lr, momentum=0.0, decay=0.0, nesterov=False)
    optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])  # here we specify the loss function

def compileModel(model,lr):
    #optimizer = SGD(learning_rate=lr, momentum=0.0, decay=0.0, nesterov=False)
    optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])  # here we specify the loss function

def main(argv):
    help_string = 'Usage: python3 bi_lstm.py --train <dataset_folder> -e <epocs>'

    parser = argparse.ArgumentParser(
        description='DDoS attacks detection with convolutional neural networks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t', '--train', nargs='+', type=str,
                        help='Start the training process')

    parser.add_argument('-e', '--epochs', default=DEFAULT_EPOCHS, type=int,
                        help='Training iterations')
    parser.add_argument('-mn','--modelname', default="BI_GRU_ATTN",type=str,
            help= 'Model Name. Available Options are SBL_A,SBLBG_A,BI_GRU_ATTN')

    parser.add_argument('-a', '--attack_net', default=None, type=str,
                        help='Subnet of the attacker (used to compute the detection accuracy)')

    parser.add_argument('-v', '--victim_net', default=None, type=str,
                        help='Subnet of the victim (used to compute the detection accuracy)')

    parser.add_argument('-p', '--predict', nargs='?', type=str,
                        help='Perform a prediction on pre-preprocessed data')

    parser.add_argument('-i', '--iterations', default=1, type=int,
                        help='Predict iterations')

    parser.add_argument('-m', '--model', type=str,
                        help='File containing the model')

    parser.add_argument('-y', '--dataset_type', default=None, type=str,
                        help='Type of the dataset. Available options are: DOS2017, DOS2018, DOS2019, SYN2020')
    
    parser.add_argument('-it','--incremental',default=False,type=bool,help="True for incremental Training")

    args = parser.parse_args()

    if os.path.isdir(OUTPUT_FOLDER) == False:
        os.mkdir(OUTPUT_FOLDER)

    if args.train is not None:
        subfolders = glob.glob(args.train[0] +"/*/")
        if len(subfolders) == 0: # for the case in which the is only one folder, and this folder is args.dataset_folder[0]
            subfolders = [args.train[0] + "/"]
        else:
            subfolders = sorted(subfolders)
        for full_path in subfolders:
            full_path = full_path.replace("//", "/")  # remove double slashes when needed
            folder = full_path.split("/")[-2]
            dataset_folder = full_path
            X_train, Y_train = load_dataset(dataset_folder + "/*" + '-train.hdf5')
            X_val, Y_val = load_dataset(dataset_folder + "/*" + '-val.hdf5')

            X_train, Y_train = shuffle(X_train, Y_train, random_state=SEED)
            X_val, Y_val = shuffle(X_val, Y_val, random_state=SEED)

            # get the time_window and the flow_len from the filename
            train_file = glob.glob(dataset_folder + "/*" + '-train.hdf5')[0]
            filename = train_file.split('/')[-1].strip()
            time_window = int(filename.split('-')[0].strip().replace('t', ''))
            max_flow_len = int(filename.split('-')[1].strip().replace('n', ''))
            dataset_name = filename.split('-')[2].strip()

            print ("\nCurrent dataset folder: ", dataset_folder)
            batch_size=1024
            model_name =  dataset_name + "-"+str(args.modelname)
            model_filename = OUTPUT_FOLDER + str(time_window) + 't-' + str(max_flow_len) + 'n-' + model_name
            input_shape=(X_train.shape[1],X_train.shape[2])

            if (args.incremental == True):
                K.clear_session()
                print("incremental training")
                model = load_model(model_filename+"-Model")
                #model = load_model(model_filename+".h5")
                print(model.summary())
                #model = load_model(model_filename+"-Model")
            else:
                model=build_model(dataset_name,str(args.modelname),input_shape=input_shape)

            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
            model_filename = OUTPUT_FOLDER + str(time_window) + 't-' + str(max_flow_len) + 'n-' + model_name
            mc = ModelCheckpoint(model_filename + '.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
            
            model.fit(X_train, Y_train, batch_size=1024, epochs=args.epochs, validation_data=(X_val, Y_val), callbacks=[es, mc])

            model.save(model_filename + '.h5')
            model.save(model_filename+"-Model")


            Y_pred_val = (model.predict(X_val) > 0.5)
            Y_true_val = Y_val.reshape((Y_val.shape[0], 1))
            f1_score_val = f1_score(Y_true_val, Y_pred_val)
            accuracy = accuracy_score(Y_true_val, Y_pred_val)

            # save best model performance on the validation set
            val_file = open(model_filename + '.csv', 'w', newline='')
            val_file.truncate(0)  # clean the file content (as we open the file in append mode)
            val_writer = csv.DictWriter(val_file, fieldnames=VAL_HEADER)
            val_writer.writeheader()
            val_file.flush()
            row = {'Model': model_name, 'Samples': Y_pred_val.shape[0], 'Accuracy': '{:05.4f}'.format(accuracy), 'F1Score': '{:05.4f}'.format(f1_score_val),
                   "Validation Set": glob.glob(dataset_folder + "/*" + '-val.hdf5')[0]}
            val_writer.writerow(row)
            val_file.close()

            print("Model path: ", model_filename)
            print("F1 Score of the best model on the validation set: ", f1_score_val)


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
            model = load_model(model_path,custom_objects={"SeqSelfAttention": SeqSelfAttention,"MultiHead":MultiHead})

            # warming up the model (necessary for the GPU)
            warm_up_file = dataset_filelist[0]
            filename = warm_up_file.split('/')[-1].strip()
            if filename_prefix in filename:
                X, Y = load_dataset(warm_up_file)
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
                        Y_pred = np.squeeze(model.predict(X, batch_size=1024) > 0.5)
                        pt1 = time.time()
                        avg_time += pt1 - pt0

                    avg_time = avg_time / iterations

                    report_results(np.squeeze(Y_true), Y_pred, packets, model_name_string, filename, avg_time,predict_writer)
                    predict_file.flush()

        predict_file.close()

def report_results(Y_true, Y_pred, packets, model_name, data_source, prediction_time, writer):
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

if __name__ == "__main__":
    main(sys.argv[1:])
