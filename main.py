import numpy as np
import matplotlib.pyplot as plt
from my_functionLib import plot_kurtosis, plot_EEG_windows, plot_Compared_EEG
import torch
from scipy.stats import kurtosis
from my_loadLib import loaddatafile, loadTrainingSet
from model import NeuralNetwork, NeuralNetwork2
import torch.nn as nn
from scipy import signal
sos = signal.butter(10, 0.5, 'hp', fs=1000, output='sos')


n = 100 # its length of the training sample
L_w = 2*n # length of windows = integral multiple of n
data_channels = loaddatafile('BCICIV_1_mat/BCICIV_calib_ds1a.mat', L_w)  # todo need load more sample for training

Testdata_channels = loaddatafile('BCICIV_1_mat/BCICIV_eval_ds1a.mat', L_w)
# Testdata_channels = data_channels[40:59]
# data_channels = data_channels[0:40]
# data_channels = loadTrainingSet(L_w)
# todo
# data_subjects =[]
# data_subjects.append(data_channels)
list_kur_allChannels = []
for data_channel in data_channels:
    list_kur = []
    for i in range(0, len(data_channel), L_w):
        list_kur.append(kurtosis(data_channel[i:i+L_w]))
    list_kur_allChannels.append(list_kur)
numbersOfWindow = 20
i_window = 10
plt.figure("kurtosisPlot")
plot_kurtosis(list_kur_allChannels[0], numbersOfWindow, i_window)
plt.figure("EEG_plot_data", figsize=(16, 4))
plot_EEG_windows(data_channels[0], numbersOfWindow, i_window,L_w)
# print(len(data_channels))
# print(len(data_channels[0]))

# delete the EEG contains OAs
th_kur = 3.5
NOE_channels =[]
X = []
for j in range(len(list_kur_allChannels)):
    list_kur_local = list_kur_allChannels[j]
    data_channel_local = data_channels[j]
    NOE_channel = []
    for i in range(len(list_kur)):
        if list_kur_local[i] < th_kur:
            NOE_channel = NOE_channel+data_channel_local[i*L_w: (i+1)*L_w]  # non-normalised, for plot comparision graph
            # print(len(NOE_channel))
            # print(type(dataset_NOE[0]))
    NOE_channels.append(NOE_channel)
plt.figure("EEG_NOE", figsize=(16, 4))
plot_EEG_windows(NOE_channels[0], numbersOfWindow, i_window,L_w)
# print(np.array(NOE_channels).shape)


# construct the model
device = "mps" if torch.has_mps else "cpu"
print(f"Using {device} device")
Batch = []
# normalised for each channel
for i in NOE_channels:
    Batch.append(nn.functional.normalize(torch.tensor(i).view(-1, 100)))

# model construction
encoder1 = NeuralNetwork().to(device)
encoder2 = NeuralNetwork2().to(device)
encoder3 = NeuralNetwork().to(device)
encoder4 = NeuralNetwork2().to(device)
loss = nn.MSELoss()
optimizater1 = torch.optim.SGD(encoder1.parameters(), lr=0.01)
optimizater2 = torch.optim.SGD(encoder2.parameters(), lr=0.01)
optimizater3 = torch.optim.SGD(encoder3.parameters(), lr=0.01)
optimizater4 = torch.optim.SGD(encoder4.parameters(), lr=0.01)

# greedy layer training
epochs_layer_training = 300 # todo 300
epochs_ft = 50 # todo 100
ls_encoder1 = []
ls_encoder2 = []
ls_encoder3 = []
ls_encoder4 = []
ls_full =[]
Batch_hidden1 = []
# training layer 1
print("training weights between input and hidden layer")
for i in range(epochs_layer_training):
    train_losses = 0
    for x in Batch:
        X = x.to(device)
        [Y_est, hidden1] = encoder1(X,device)
        l = loss(X, Y_est)
        # add to train_losses
        l.backward()
        optimizater1.step()
        loss_cpu = l.detach().cpu().item()
        train_losses += loss_cpu
        optimizater1.zero_grad()
    ls_encoder1.append(train_losses)
    # print(f"epochs: {i}, cost is {train_losses}")
# for para in encoder1.parameters():  # todo need to print the parameter with details name
#     print(para.name, para.data)
# training layer 2
print("training weights between hidden1 and hidden2 layer")
for i in range(epochs_layer_training):
    train_losses = 0
    for x in Batch:
        X = x.to(device)
        [Y_est, input] = encoder1(X,device)
        [Y_est, hidden2] = encoder2(input, device)
        l = loss(input, Y_est)
        # add to train_losses
        l.backward()
        optimizater2.step()
        loss_cpu = l.detach().cpu().item()
        train_losses += loss_cpu
        optimizater2.zero_grad()
    ls_encoder2.append(train_losses)
    # print(f"epochs: {i}, cost is {train_losses}")


# training layer 3
print("training weights between hidden2 and hidden3 layer")
for i in range(epochs_layer_training):
    train_losses = 0
    for x in Batch:
        X = x.to(device)
        [Y_est, input] = encoder1(X,device)
        [Y_est, input] = encoder2(input, device)
        [Y_est, hidden3] = encoder3(input, device)
        l = loss(input, Y_est)
        # add to train_losses
        l.backward()
        optimizater3.step()
        loss_cpu = l.detach().cpu().item()
        train_losses += loss_cpu
        optimizater3.zero_grad()
    ls_encoder3.append(train_losses)
    # print(f"epochs: {i}, cost is {train_losses}")

# training layer 4 fiction a layer
print("training weights between hidden3 and output layer")
for i in range(epochs_layer_training):
    train_losses = 0
    for x in Batch:
        X = x.to(device)
        [Y_est, input] = encoder1(X,device)
        [Y_est, input] = encoder2(input, device)
        [Y_est, input] = encoder3(input, device)
        [Y_est, hidden4] = encoder4(input, device)
        l = loss(input, Y_est)
        # add to train_losses
        l.backward()
        optimizater4.step()
        loss_cpu = l.detach().cpu().item()
        train_losses += loss_cpu
        optimizater4.zero_grad()
    ls_encoder4.append(train_losses)
    # print(f"epochs: {i}, cost is {train_losses}")
# for para in encoder1.parameters():  # todo need to print the parameter with details name
#     print(para.name, para.data)


# fine-tuning
print("fine_tuning")
for i in range(epochs_ft):
    train_losses = 0
    for x in Batch:
        X = x.to(device)
        [temp, hidden1] = encoder1(X,device)
        [temp, hidden2] = encoder2(hidden1, device)
        [temp, hidden3] = encoder3(hidden2, device)
        [temp, output] = encoder4(hidden3, device)

        l = loss(X, output)
        # add to train_losses
        l.backward()
        optimizater4.step()
        optimizater3.step()
        optimizater2.step()
        optimizater1.step()

        loss_cpu = l.detach().cpu().item()
        train_losses += loss_cpu
        optimizater4.zero_grad()
        optimizater3.zero_grad()
        optimizater2.zero_grad()
        optimizater1.zero_grad()
    ls_full.append(train_losses)
    print(f"fine tuning epochs: {i}, cost is {train_losses}")

plt.figure("comparision of containminated and reconstructed EEG", figsize=(16, 4))
plot_Compared_EEG(X.to('cpu').detach().numpy(), output.to('cpu').detach().numpy(), numbersOfWindow, i_window)



ls_epoch = [i for i in range(epochs_layer_training)]
plt.figure("layer cost function over epochs")
fig, a = plt.subplots(2,2)
x = np.arange(1,5)
a[0][0].plot(ls_epoch, ls_encoder1)
a[0][0].set_title('training loss for layer 1')
a[0][1].plot(ls_epoch, ls_encoder2)
a[0][1].set_title('training loss for layer 2')
a[1][0].plot(ls_epoch, ls_encoder3)
a[1][0].set_title('training loss for layer 3')
a[1][1].plot(ls_epoch, ls_encoder4)
a[1][1].set_title('training loss for layer 4')

plt.figure("full network training cost over epochs")
ls_epoch = [i for i in range(epochs_ft)]
plt.plot(ls_epoch, ls_full)

### Testing
from my_functionLib import combine_channel
# todo: add step to save and reload the model.
# Testdata_channels = loaddatafile('BCICIV_1_mat/BCICIV_calib_ds1b.mat', L_w)
# data_test = combine_channel(list_TestingDataInWindow)
# todo need load more sample for testing
# select a NOE as reference
crt_channels = []
ctn_channels = []
RMSE_otp = []
RMSE_inp = []
RMSE_crt = []
RMSE_NOE = []
for test_data in Testdata_channels:
    list_ctn = []
    list_noe = []
    index_NOE = []
    index_ctd = []
    list_kur_test = []
    for index in range(0, len(test_data), L_w):
        j = test_data[index:index+L_w]
        kur = kurtosis(j)
        list_kur_test.append(kurtosis(kur))
        if kur > th_kur:
            index_ctd.append(int(index/100))
            index_ctd.append(int(index/100)+1)
        else:
            index_NOE.append(int(index/100))
            index_NOE.append(int(index/100) + 1)
            list_noe.extend(j[0:100])
            list_noe.extend(j[100:200])
        list_ctn.append(j[0:100])
        list_ctn.append(j[100:200])
    RMSE_inp.extend(list_noe)
    from my_functionLib import standardlization
    # standarlizaiton
    # initialise the reference
    reference = test_data[index_NOE[0]:index_NOE[0]+100]
    # reference = nn.functional.normalize(torch.tensor(reference).view(-1, 100)).tolist()[0]
    X_std, list_min = standardlization(reference, list_ctn)
    X_std = torch.stack(X_std).to(device)

    non, vector_hidden1 = encoder1(X_std, device)
    non, vector_hidden2 = encoder2(vector_hidden1, device)
    non, vector_hidden3 = encoder3(vector_hidden2, device)
    non, Y_opt = encoder4(vector_hidden3, device)
    # reconstruct
    EEG_opt = Y_opt.to('cpu').detach().numpy()
    list_crt = []
    SN_min = min(reference)
    SN_max = max(reference)
    temp = []
    for i in range(len(EEG_opt)):
        temp_min = list_min[i]
        crt = []
        for opt_k in EEG_opt[i]:
            crt_k = (opt_k + temp_min)*(SN_max -SN_min) +SN_min
            crt.append(crt_k)
        temp.extend(crt)
        list_crt.append(crt)
    crt_channels.append(list_crt)
    ctn_channels.append(list_ctn)
    # extract the corrected segment
    NOE = list_ctn[index_NOE[0]]
    for i in index_ctd:
        a = list_crt[i]
        # a = signal.sosfilt(sos, a)
        RMSE_crt.extend(a)
        RMSE_NOE.extend(NOE)
    for i in index_NOE:
        a = list_crt[i]
        # a = signal.sosfilt(sos, a)
        RMSE_otp.extend(a)

# split into reconstruction and
from my_functionLib import plot_Compared_EEG1
plt.figure('testing')
print(len(ctn_channels))
print(len(crt_channels))
plot_Compared_EEG1(ctn_channels[1], crt_channels[1], numbersOfWindow, i_window)

# quantifying
diff_recon = np.subtract(np.array(RMSE_otp), np.array(RMSE_inp))
diff_removal = np.subtract(np.array(RMSE_crt), np.array(RMSE_NOE))
sum_recon = np.square(diff_recon.mean())
sum_removal = np.square(diff_removal.mean())
RMSE_recon = np.sqrt(sum_recon)
RMSE_removal = np.sqrt(sum_removal)
print(f"the RMSE for reconstuction is {RMSE_recon}")
print(f"the RMSE for removal is {RMSE_removal}")

plt.show()
print("finished")
