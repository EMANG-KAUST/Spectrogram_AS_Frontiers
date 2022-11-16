%==========================================================================
%   SCSA parameter selection
%   Author: Juan M. Vargas
%   E-mail: jm.manuelvg@gmail.com
%   June 30th, 2022
%==========================================================================
clear all
clc


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load folder and Dataset
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

w_type="Hamming"; %Select window type from : Hamming or Kaiser

overlap=0; % Overlaping percentage: Hamming: 0, 60, 95 and Kaiser: 0 , 61, 75 

gm=[0,2,4,6,8,10,12,14,16,18,20]; % Gamma values for SCSA 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Folder creation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filen=strcat('./Results/SCSA_tune/',wav,'_',sig_n,'_Spectrogram_selec_',w_type,'_s=',num2str(SNR)); % Full name of the folder

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
% Create spectrogram
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[spec_img]=CreateSpectrogram(sig_nf,"Spectrogram abs",w_type,2,499,overlap,0);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCSA computation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for im=1:1%size(spec_img,3)
   
img_or=spec_img(:,:,im); % Load spectrogram

% Find h_min
beta=1;
max_r= max(img_or,[],2); % Find maximum values of rows
max_c= max(img_or,[],1); % Find maximum values of columns 
h_min_row=beta*(1/pi).*sqrt(double(max_r)); % compute the h_min per rows (hr_min)
h_min_column=beta*(1/pi).*sqrt(double(max_c)); % compute the h_min per columns (hc_min)
h_min=max((h_min_row+h_min_column.')./2,[],'all');  % Find h_min for all the imge

for g=1:length(gm)        
h_act(im)=h_min;
gamma=gm(g);
fe=1;
img_s=double(img_or);
[img_scsa,psiy,psix,v1,NY,NX]=SCSA_2D1D(img_s,h_act(im),fe,gamma); % compute 2D-SCSA
PSNR(im,g)=psnr(img_scsa,img_s); % Compute PSNR
SSIM(im,g)=ssim(img_scsa,img_s); % Compute SSIM
end
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

writematrix(PSNR,strcat(filen+'/PSNR.csv'))
writematrix(SSIM,strcat(filen+'/SSIM.csv'))
writematrix(h_act,strcat(filen+'/h.csv'))
writematrix(gm,strcat(filen+'/gamma.csv'))