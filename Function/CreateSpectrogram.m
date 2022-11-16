function ps_img=CreateSpectrogram(signals,img_type,type,window_s,s1,overlap,alpha)

switch(type)

    case "Hamming"
    % Hamming window
    ps_img=spectrogram_PPG(signals,img_type,"Hamming",window_s,s1,overlap,alpha);
    case "Kaiser"
    % Kaiser window
    ps_img=spectrogram_PPG(signals,img_type,"Kaiser",window_s,s1,overlap,alpha);
   
end

