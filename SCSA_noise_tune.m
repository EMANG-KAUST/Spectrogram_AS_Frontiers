%==========================================================================
%   SCSA parameter selection for Noisy data 
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

filen=strcat('./Results/SCSA_tune_noise/',wav,'_',sig_n,'_Spectrogram_selec_',w_type,'_s=',num2str(SNR)); % Full name of the folder

mkdir(filen) % Create folder


%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load signals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:4374
sig_or=data.waves.PPG_Brachial{1,i}; % Load Signal
sig=awgn(sig_or,str2double(SNR),1); % Add noise

sig_nf{i,1}=(sig_or-min(sig_or))/(max(sig_or)-min(sig_or)); % Normalize signal
sig_ns{i,1}=(sig-min(sig))/(max(sig)-min(sig)); % Normalize noisy signal

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Spectrogram creation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[spec_img_ref]=CreateSpectrogram(sig_nf,"Spectrogram abs",w_type,2,499,overlap,0); % free-noise spectrogram
[spec_img]=CreateSpectrogram(sig_ns,"Spectrogram abs",w_type,2,499,overlap,0); % Noise spectrogram


%subplot(1,2,1),imshow(spec_img_ref),title('Original')
%subplot(1,2,2),imshow(spec_img),title('Noisy')

for im=1:1%size(spec_img,3)
   
img_or=spec_img(:,:,im); % Load spectrogram

% Find h_min
beta=1;
max_r= max(img_or,[],2); % Find maximum values of rows
max_c= max(img_or,[],1); % Find maximum values of columns 
h_min_row=beta*(1/pi).*sqrt(double(max_r)); % compute the h_min per rows (hr_min)
h_min_column=beta*(1/pi).*sqrt(double(max_c)); % compute the h_min per columns (hc_min)
h_min=max((h_min_row+h_min_column.')./2,[],'all');  % Find h_min for all the imge
    
h_vec=[h_min,0.2,0.5,0.8,1,1.2,1.4]; % h values 
for h=1:size(h_vec,2)
for g=1:length(gm)        
h_act(h)=h_vec(h);
gamma=gm(g);
fe=1;
img_s=double(img_or);
[img_scsa,psiy,psix,v1,NY,NX]=SCSA_2D1D(img_s,h_act(h),fe,gamma); % 2D-SCSA computing
PSNR(h,g)=psnr(img_scsa,spec_img_ref(:,:,im)); % PSRN computing
SSIM(h,g)=ssim(img_scsa,spec_img_ref(:,:,im)); % SSIM computing
end
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