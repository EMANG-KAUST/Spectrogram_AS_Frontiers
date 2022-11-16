%==========================================================================
%   Spectrogram parameter selection
%   Author: Juan M. Vargas
%   E-mail: jm.manuelvg@gmail.com
%   June 22th, 2022
%==========================================================================
clear all
clc



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Load folder and Dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath './Function'  % add function folder
addpath './Data'  % add data folder
load('pwdb_data.mat') % Load in-silico dataset

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Select Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sig_n="Radial"; % Signal location : Radial , Brachial and Digital

wav='BP'; % Signal type : BP or PPG

SNR="20"; % Noise level : PPG: 65, 45 and 30  and BP: 20, 10, 5

window_type="Hamming"; %Select window type from : Hamming or Kaiser

overlap=0; % Overlaping percentage: Hamming: 0, 60, 95 and Kaiser: 0 , 61, 75 

alpha=0; % Alpha parameter of Kaiser window



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Folder creation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filen=strcat('./Results/spectrogram-selection/',wav,'_',sig_n,'_Spectrogram_selec_',window_type,"_a=",num2str(alpha),"_o=",num2str(overlap),'_s=',num2str(SNR)); % Full name of the folder

mkdir(filen) % Create folder


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load signals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:4374

sig=data.waves.P_Radial{1,i}; % Load signal

if strcmp(SNR,"no")==0
    sig=awgn(sig,str2double(SNR)); % Add Gaussian White noise

end

sig_nf{i,1}=(sig-min(sig))/(max(sig)-min(sig)); % Normalize signal


end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create spectrogram and compute Q-metrics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% window size: 20
[Spectrogram_abs_Q_t_20_,Spectrogram_abs_Q_f_20_,Spectrogram_abs_Q_tf_20_,spec_img_20]=spectrogram_metric(sig_nf,"Spectrogram abs",window_type,23,39,overlap,alpha);

% window size: 50
[Spectrogram_abs_Q_t_50_,Spectrogram_abs_Q_f_50_,Spectrogram_abs_Q_tf_50_,spec_img_50]=spectrogram_metric(sig_nf,"Spectrogram abs",window_type,10,99,overlap,alpha);


% window size: 100
[Spectrogram_abs_Q_t_100_,Spectrogram_abs_Q_f_100_,Spectrogram_abs_Q_tf_100_,spec_img_100]=spectrogram_metric(sig_nf,"Spectrogram abs",window_type,5,199,overlap,alpha);


% window size: 166
[Spectrogram_abs_Q_t_166_,Spectrogram_abs_Q_f_166_,Spectrogram_abs_Q_tf_166_,spec_img_166]=spectrogram_metric(sig_nf,"Spectrogram abs",window_type,3,331,overlap,alpha);


% window size: 250
[Spectrogram_abs_Q_t_250_,Spectrogram_abs_Q_f_250_,Spectrogram_abs_Q_tf_250_,spec_img_250]=spectrogram_metric(sig_nf,"Spectrogram abs",window_type,2,499,overlap,alpha);

Qt_Spectrogram=[Spectrogram_abs_Q_t_20_;Spectrogram_abs_Q_t_50_;Spectrogram_abs_Q_t_100_;Spectrogram_abs_Q_t_166_;Spectrogram_abs_Q_t_250_];
Qf_Spectrogram=[Spectrogram_abs_Q_f_20_;Spectrogram_abs_Q_f_50_;Spectrogram_abs_Q_f_100_;Spectrogram_abs_Q_f_166_;Spectrogram_abs_Q_f_250_];
Qtf_Spectrogram=[Spectrogram_abs_Q_tf_20_;Spectrogram_abs_Q_tf_50_;Spectrogram_abs_Q_tf_100_;Spectrogram_abs_Q_tf_166_;Spectrogram_abs_Q_tf_250_];

Q_metrics=[Qt_Spectrogram,Qf_Spectrogram,Qtf_Spectrogram];

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

table_Q_metric=array2table(Q_metrics,'RowNames',{'20','50','100','166','250'},'VariableNames',{'Qt','Qf','Qtf'});

writetable(table_Q_metric,strcat(filen+strcat("/Table Q metric ",window_type," spectrogram_SNR=",SNR,".csv")))

