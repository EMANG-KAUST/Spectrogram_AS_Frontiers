%==========================================================================
%   SCSA features extraction
%   Author: Juan M. Vargas
%   E-mail: jm.manuelvg@gmail.com
%   July 15th, 2022
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

SNR="no"; % Noise level : PPG: 65, 45 and 30  and BP: 20, 10, 5

w_type="Hamming"; %Select window type from : Hamming or Kaiser

overlap=0; % Overlaping percentage: Hamming: 0, 60, 95 and Kaiser: 0 , 61, 75 

models='SCSA'; % Features selection method

gm=4; % gamma parameter
fe=1

vec_seeds=[5]; % Seed value 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Folder creation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filen=strcat('./Results/ML/',models,'_',wav,'_',sig_n,'_SNR=',SNR) % Full name of the folder

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

tab=struct2table(data.haemods);
cfpwv=tab.('PWV_cf');


%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Split dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for j=1:size(vec_seeds,2)
rng(vec_seeds(j)) % different arbitrary choice

% randomly select indexes to split data into 70% 
% training set, 0% validation set and 30% test set.
[train_idx(j,:), ~, test_idx(j,:)] = dividerand(length(sig_nf), 0.7, 0,0.3);
% slice training data with train indexes 
x_train = sig_nf(train_idx(j,:), :);

x_test=sig_nf(test_idx(j,:), :);


% select test data
y_train(:,j) = cfpwv(train_idx(j,:), :);
y_test(:,j) = cfpwv(test_idx(j,:), :);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Spectrogram creation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[spec_img]=CreateSpectrogram(x_train,"Spectrogram abs",w_type,2,499,overlap,0);

[spec_img_test]=CreateSpectrogram(x_test,"Spectrogram abs",w_type,2,499,overlap,0);


for s=1:size(x_train,1)

strcat('Running the Train signal # ',num2str(s))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the SCSA-based features: TRAIN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img_or=spec_img(:,:,s);

    fe=1;
    gm=4;
    beta=1;
    max_r= max(img_or,[],2);
    max_c= max(img_or,[],1);
    h_min_row=beta*(1/pi).*sqrt(double(max_r));
    h_min_column=beta*(1/pi).*sqrt(double(max_c));
    h_min=max((h_min_row+h_min_column.')./2,[],'all');
    h=h_min;
    [img_scsa,psiy,psix,v1,kapx,kapy,Nx,Ny]=SCSA_2D1D(img_or,h,fe,gm); % Apply 2D1D-SCSA

    




% %%%%%%%%%%%%%%%%%%%% Extraction for all rows %%%%%%%%%%%%%%%%%%%%%%%%%%%

% Stadistical based
mean_rows(:,s)=mean(mean(kapx,2));
std_rows(:,s)=mean(std(kapx,0,2));

% Invariants
INV1_rows(:,s)=mean(4*h*sum(kapx,2));
INV2_rows(:,s)=mean(((16*h)/3) *sum(kapx.^3,2));
INV3_rows(:,s)=mean(((256*h)/7) *sum(kapx.^7,2));

% First three eigen-values

First_rows(:,s)=mean(kapx(:,1));
Second_rows(:,s)=mean(kapx(:,2));
Third_rows(:,s)=mean(kapx(:,3));

% Number of eigen values
N_row(:,s)=mean(Nx,'all');

% Squared-eigen values 
K1sq_row(:,s)=(mean(kapx(:,1)))^2;
K2sq_row(:,s)=(mean(kapx(:,2)))^2;
K3sq_row(:,s)=(mean(kapx(:,3)))^2;

% Ratio
K1r_row(:,s)=mean(kapx(:,1))/h;
K1m_row(:,s)=median(kapx(:,1))/h;

%%%%%%%%%%%%%%%%%%%% Extraction for all columns %%%%%%%%%%%%%%%%%%%%%%%%%%%

% Stadistical based
mean_columns(:,s)=mean(mean(kapy,2));
std_columns(:,s)=mean(std(kapy,0,2));

% Invariants
INV1_columns(:,s)=mean(4*h*sum(kapy,2));
INV2_columns(:,s)=mean(((16*h)/3) *sum(kapy.^3,2));
INV3_columns(:,s)=mean(((256*h)/7) *sum(kapy.^7,2));

% First three eigen-values


First_columns(:,s)=mean(kapy(:,1));
Second_columns(:,s)=mean(kapy(:,2));
Third_columns(:,s)=mean(kapy(:,3));

% Number of eigenvalues
N_column(:,s)=mean(Ny,'all');

% Squared-eigenvalues
K1sq_column(:,s)=(mean(kapy(:,1)))^2;
K2sq_column(:,s)=(mean(kapy(:,2)))^2;
K3sq_column(:,s)=(mean(kapy(:,3)))^2;

% Ratio
K1r_column(:,s)=mean(kapy(:,1))/h;
K1m_column(:,s)=median(kapy(:,1))/h;

%%%%%%%%%%%%%%%%%%%% Extraction for sum %%%%%%%%%%%%%%%%%%%%%%%%%%%

kapsum=kapx+kapy;

% Stadistical based
mean_sums(:,s)=mean(mean(kapsum,2));
std_sums(:,s)=mean(std(kapsum,0,2));

% Invariants
INV1_sums(:,s)=mean(4*h*sum(kapsum,2));
INV2_sums(:,s)=mean(((16*h)/3) *sum(kapsum.^3,2));
INV3_sums(:,s)=mean(((256*h)/7) *sum(kapsum.^7,2));

% First three eigen-values


First_sums(:,s)=mean(kapsum(:,1));
Second_sums(:,s)=mean(kapsum(:,2));
Third_sums(:,s)=mean(kapsum(:,3));

% Number of eigenvalues
N_sum(:,s)=mean(Nx+Ny,'all');

% Squared-eigenvalues
K1sq_sum(:,s)=(mean(kapsum(:,1)))^2;
K2sq_sum(:,s)=(mean(kapsum(:,2)))^2;
K3sq_sum(:,s)=(mean(kapsum(:,3)))^2;

% Ratio
K1r_sum(:,s)=mean(kapsum(:,1))/h;
K1m_sum(:,s)=median(kapsum(:,1))/h;
end

%%%%%%%%%%%%%%%%%%%%%% Features based  on SCSA eigenvalues %%%%%%%%%%%%%%%%%%%

mf_c=[K1sq_column',K2sq_column',K3sq_column',K1r_column',K1m_column',N_column'];
mf_r=[K1sq_row',K2sq_row',K3sq_row',K1r_row',K1m_row',N_row'];
mf_s=[K1sq_sum',K2sq_sum',K3sq_sum',K1r_sum',K1m_sum',N_sum'];

%%%%%%%%%%%%%%%%% Features based on SCSA statistical moments %%%%%%%%%%%%
       
ms_c=[mean_columns;std_columns;INV1_columns;INV2_columns;INV3_columns;First_columns]';
ms_r=[mean_rows;std_rows;INV1_rows;INV2_rows;INV3_rows;First_rows]';
ms_s=[mean_sums;std_sums;INV1_sums;INV2_sums;INV3_sums;First_sums]';

%%%%%%%%%%%%%%%%%%%%%% Features matrix  %%%%%%%%%%%%%%%%%%%

feature_train(:,:,j)=cat(2,mf_c,mf_r,mf_s,ms_c,ms_r,ms_s);




[idx(j,:),scores(j,:)] = fsrmrmr(feature_train(:,:,j),y_train);



for s_test=1:size(x_test,1)
strcat('Running the Test signal # ',num2str(s_test))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the SCSA-based features:TEST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img_or=spec_img_test(:,:,s_test);

    fe=1;
    gm=4;
    beta=1;
    max_r= max(img_or,[],2);
    max_c= max(img_or,[],1);
    h_min_row=beta*(1/pi).*sqrt(double(max_r));
    h_min_column=beta*(1/pi).*sqrt(double(max_c));
    h_min=max((h_min_row+h_min_column.')./2,[],'all');
    h=h_min;
    [img_scsa_t,psiy_t,psix_t,v1_t,kapx_t,kapy_t,Nx_t,Ny_t]=SCSA_2D1D(img_or,h,fe,gm); % Apply 2D1D-SCSA





    





% %%%%%%%%%%%%%%%%%%%% Extraction for all rows %%%%%%%%%%%%%%%%%%%%%%%%%%%

% Stadistical based
mean_rows_t(:,s_test)=mean(mean(kapx_t,2));
std_rows_t(:,s_test)=mean(std(kapx_t,0,2));
median_rows_t(:,s_test)=mean(median(kapx_t,2));

% Invariants
INV1_rows_t(:,s_test)=mean(4*h*sum(kapx_t,2));
INV2_rows_t(:,s_test)=mean(((16*h)/3) *sum(kapx_t.^3,2));
INV3_rows_t(:,s_test)=mean(((256*h)/7) *sum(kapx_t.^7,2));

% First three eigen-values

First_rows_t(:,s_test)=mean(kapx_t(:,1));
Second_rows_t(:,s_test)=mean(kapx_t(:,2));
Third_rows_t(:,s_test)=mean(kapx_t(:,3));

% Number of eigen values
N_row_t(:,s_test)=mean(Nx_t,'all');

% Squared-eigen values 
K1sq_row_t(:,s_test)=(mean(kapx_t(:,1)))^2;
K2sq_row_t(:,s_test)=(mean(kapx_t(:,2)))^2;
K3sq_row_t(:,s_test)=(mean(kapx_t(:,3)))^2;

% Ratio
K1r_row_t(:,s_test)=mean(kapx_t(:,1))/h;
K1m_row_t(:,s_test)=median(kapx_t(:,1))/h;

%%%%%%%%%%%%%%%%%%%% Extraction for all columns %%%%%%%%%%%%%%%%%%%%%%%%%%%

% Stadistical based
mean_columns_t(:,s_test)=mean(mean(kapy_t,2));
std_columns_t(:,s_test)=mean(std(kapy_t,0,2));
median_columns_t(:,s_test)=mean(median(kapy_t,2));

% Invariants
INV1_columns_t(:,s_test)=mean(4*h*sum(kapy_t,2));
INV2_columns_t(:,s_test)=mean(((16*h)/3) *sum(kapy_t.^3,2));
INV3_columns_t(:,s_test)=mean(((256*h)/7) *sum(kapy_t.^7,2));

% First three eigen-values


First_columns_t(:,s_test)=mean(kapy_t(:,1));
Second_columns_t(:,s_test)=mean(kapy_t(:,2));
Third_columns_t(:,s_test)=mean(kapy_t(:,3));

% Number of eigenvalues
N_column_t(:,s_test)=mean(Ny_t,'all');

% Squared-eigenvalues
K1sq_column_t(:,s_test)=(mean(kapy_t(:,1)))^2;
K2sq_column_t(:,s_test)=(mean(kapy_t(:,2)))^2;
K3sq_column_t(:,s_test)=(mean(kapy_t(:,3)))^2;

% Ratio
K1r_column_t(:,s_test)=mean(kapy_t(:,1))/h;
K1m_column_t(:,s_test)=median(kapy_t(:,1))/h;

%%%%%%%%%%%%%%%%%%%% Extraction for sum %%%%%%%%%%%%%%%%%%%%%%%%%%%

kapsum_t=kapx_t+kapy_t;

% Stadistical based
mean_sums_t(:,s_test)=mean(mean(kapsum_t,2));
std_sums_t(:,s_test)=mean(std(kapsum_t,0,2));
median_sums_t(:,s_test)=mean(median(kapsum_t,2));

% Invariants
INV1_sums_t(:,s_test)=mean(4*h*sum(kapsum_t,2));
INV2_sums_t(:,s_test)=mean(((16*h)/3) *sum(kapsum_t.^3,2));
INV3_sums_t(:,s_test)=mean(((256*h)/7) *sum(kapsum_t.^7,2));

% First three eigen-values


First_sums_t(:,s_test)=mean(kapsum_t(:,1));
Second_sums_t(:,s_test)=mean(kapsum_t(:,2));
Third_sums_t(:,s_test)=mean(kapsum_t(:,3));

% Number of eigenvalues
N_sum_t(:,s_test)=mean(Nx_t+Ny_t);

% Squared-eigenvalues
K1sq_sum_t(:,s_test)=(mean(kapsum_t(:,1)))^2;
K2sq_sum_t(:,s_test)=(mean(kapsum_t(:,2)))^2;
K3sq_sum_t(:,s_test)=(mean(kapsum_t(:,3)))^2;

% Ratio
K1r_sum_t(:,s_test)=mean(kapsum_t(:,1))/h;
K1m_sum_t(:,s_test)=median(kapsum_t(:,1))/h;
end

%%%%%%%%%%%%%%%%%%%%%% Features based  on SCSA eigenvalues %%%%%%%%%%%%%%%%%%%

mf_c_t=[K1sq_column_t',K2sq_column_t',K3sq_column_t',K1r_column_t',K1m_column_t',N_column_t'];
mf_r_t=[K1sq_row_t',K2sq_row_t',K3sq_row_t',K1r_row_t',K1m_row_t',N_row_t'];
mf_s_t=[K1sq_sum_t',K2sq_sum_t',K3sq_sum_t',K1r_sum_t',K1m_sum_t',N_sum_t'];

%%%%%%%%%%%%%%%%% Features based on SCSA statistical moments %%%%%%%%%%%%
       
ms_c_t=[mean_columns_t;std_columns_t;INV1_columns_t;INV2_columns_t;INV3_columns_t;First_columns_t]';
ms_r_t=[mean_rows_t;std_rows_t;INV1_rows_t;INV2_rows_t;INV3_rows_t;First_rows_t]';
ms_s_t=[mean_sums_t;std_sums_t;INV1_sums_t;INV2_sums_t;INV3_sums_t;First_sums_t]';

%%%%%%%%%%%%%%%%%%%%%% Features matrix  %%%%%%%%%%%%%%%%%%%

feature_test(:,:,j)=cat(2,mf_c_t,mf_r_t,mf_s_t,ms_c_t,ms_r_t,ms_s_t);


end



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

writematrix(feature_train,strcat(filen,'/','features_train_all.csv'))
writematrix(feature_test,strcat(filen,'/','features_test_all.csv'))
save(strcat(filen,'/','features_train_all.mat'),'feature_train')
save(strcat(filen,'/','features_test_all.mat'),'feature_test')
writematrix(idx,strcat(filen+'/indx_features.csv'))
writematrix(scores,strcat(filen+'/scores_features.csv'))
best_idx=mode(idx,1);
writematrix(best_idx,strcat(filen,'/','best_feature_indx.csv'))
save(strcat(filen,'/','y_train.mat'),'y_train')
save(strcat(filen,'/','y_test.mat'),'y_test')

