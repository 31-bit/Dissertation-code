import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import FixedFormatter
import torch

def combine_channel (data):
    result = []
    for i in data:
        result += i
    return result
# def kl_divergence(rho, rho_hat):
#     rho_hat = torch.mean(F.sigmoid(rho_hat), 1) # sigmoid because we need the probability distributions
#     rho = torch.tensor([rho] * len(rho_hat)).to(device)
#     return torch.sum(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))
# # define the sparse loss function
# def sparse_loss(rho, images):
#     values = images
#     loss = 0
#     for i in range(len(model.children)):
#         values = model.children[i](values)
#         loss += kl_divergence(rho, values)
#     return loss

def standardlization(reference,list_ctn):
    ### input contaiminated signal(ctn, data structure: window list in list)
    # return standardlization signal(std, data structure tensor in list)
    SN_min = min(reference)
    SN_max = max(reference)
    X_std = []
    list_min = []
    for i in list_ctn:
        temp_list = []
        for ctn in i:
            temp_list.append((ctn - SN_min) / (SN_max - SN_min))
        #     print(len(temp_list))
        min_temp = min(temp_list)
        list_min.append(min_temp)
        new_window = [k - min_temp for k in temp_list]
        X_std.append(torch.tensor(new_window))
    return X_std, list_min

def plot_kurtosis(list_kur,numbersOfWindow, i_window):
    index_list = [i for i in range(numbersOfWindow)]
    plt.stem(index_list, list_kur[i_window:i_window+numbersOfWindow])
    plt.title('the kurtosis plot for each window')
    plt.xlabel('window number')
    plt.ylabel('kurtosis value')
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()

    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数


def plot_EEG_windows(data, numbersOfWindow, i_window, L_w):
    plt.plot(data[i_window*L_w:(i_window+numbersOfWindow)*L_w], 'r--', label='type1')
    plt.title('A typical contaminated EEG signal for 20s.')
    plt.xlabel('sampling point')
    plt.ylabel('Amplitude/uV')
    x_major_locator = MultipleLocator(L_w)
    ax = plt.gca()
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright =False)
    ax.xaxis.set_major_locator(x_major_locator)
    # todo change axes percision to seconds


def plot_Compared_EEG(contaminated_EEG, reconstruct_EEG, numbersOfWindow, i_window):
    windows1 = contaminated_EEG[i_window:i_window + numbersOfWindow]
    windows2 = reconstruct_EEG[i_window:i_window + numbersOfWindow]
    plot_con = []
    plot_re = []
    for i in range(len(windows1)):
        plot_con.extend(np.ndarray.tolist(windows1[i]))
        plot_re.extend(np.ndarray.tolist(windows2[i]))
    plt.plot(plot_con, 'r--', label='contaminated EEG')
    plt.plot(plot_re, 'g--', label='Reconstructed EEG')
    plt.xlabel('seconds/s')
    plt.ylabel('Signal Amplitude/uV')
    x_major_locator = MultipleLocator(len(contaminated_EEG[0]))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    m = [i//100 for i in range(0, 2100, 100)]
    majors = [''] + m
    ax.xaxis.set_major_formatter(FixedFormatter(majors))
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend(loc=1)
    # 把x轴的主刻度设置为1的倍数
def plot_Compared_EEG1(contaminated_EEG, reconstruct_EEG, numbersOfWindow, i_window):
    windows1 = contaminated_EEG[i_window:i_window + numbersOfWindow]
    windows2 = reconstruct_EEG[i_window:i_window + numbersOfWindow]
    plot_con = []
    plot_re = []
    for i in range(len(windows1)):
        plot_con.extend((windows1[i]))
        plot_re.extend(windows2[i])
    plt.plot(plot_con, 'r--', label='contaminated EEG')
    plt.plot(plot_re, 'g--', label='Reconstructed EEG')
    plt.xlabel('seconds/s')
    plt.ylabel('Signal Amplitude/uV')
    x_major_locator = MultipleLocator(len(contaminated_EEG[0]))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    m = [i//100 for i in range(0, 2100, 100)]
    majors = [''] + m
    ax.xaxis.set_major_formatter(FixedFormatter(majors))
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend(loc=1)

# def plot_Reconstruction(title="",data,numbersOfWindow):
#     plt.figure(title)
#
#     for i in list_ctn[index_NOE[1]:index_NOE[1] + numbersOfWindow]:
#         el = np.ndarray.tolist(i)
#         EEG_plot_data1.extend(el)
#     plt.plot(EEG_plot_data1, 'r--', label='type1')
# def plot_

