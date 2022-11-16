function [spec_img_Q_t,spec_img_Q_f,spec_img_Q_tf,spec_img]=spectrogram_metric(sig_nf,img_type,w_type,window_s,s1,overlap,alpha)

[spec_img]=CreateSpectrogram(sig_nf,img_type,w_type,window_s,s1,overlap,alpha);
        
spec_img_mean_t=mean(spec_img,1);
spec_img_mean_f=mean(spec_img,2);

spec_img_std_t=std(spec_img,0,1);
spec_img_std_f=std(spec_img,0,2);

spec_img_c_t=spec_img_mean_t./spec_img_std_t;
spec_img_c_f=spec_img_mean_f./spec_img_std_f;


spec_img_Q_t=mean(mean(spec_img_c_t,2),"all");
spec_img_Q_f=mean(mean(spec_img_c_f,1),"all");
spec_img_Q_tf=mean(mean(spec_img_Q_f.*spec_img_Q_t),"all");








end