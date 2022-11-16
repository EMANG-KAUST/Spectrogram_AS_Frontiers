function [ps_img]=spectrogram_PPG(signals,img_type,window,s0,s1,overlap,alpha)




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create spectrograms (Train and test)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% Train spectrograms 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for p=1:size(signals,1)

sigs=signals{p,1};
sigs=sigs.';

t0=0:1/500:(length(sigs)-1)/500;
tend=t0(end);
tf=linspace(0,tend,500);
vq1= interp1(t0,sigs,tf);  % Umsampling signal

wl=s0;



ov_n=round(overlap*wl);
wl=round(wl+ov_n);

if window=="Hamming"
% Hamming windows
[a,~,~,psd]=spectrogram(vq1,hamming(wl),ov_n,s1,500); % Create spectrogram 50 x 50
elseif window=="Kaiser"
% Kaiser windows
[a,~,~,psd]=spectrogram(vq1,kaiser(wl,alpha),ov_n,s1,500); % Create spectrogram 50 x 50

end

if img_type=="PSD"
ps=psd;
elseif img_type=="Spectrogram abs"
ps=abs(a);
end

ps_img(:,:,p)=ps;

end





