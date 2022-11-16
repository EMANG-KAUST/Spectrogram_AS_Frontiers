function [image_out] = Law_mask(file_name, filter_type, window_size, statistic_type, normalization_type)

if filter_type(2)=='3'
L3 = [1 2 1]; E3 = [-1 0 1]; S3 = [-1 2 -1]; L3L3mask = L3' * L3;
switch filter_type
case 'L3L3'
filter_mask = L3' * L3;
case 'E3E3'
filter_mask = E3' * E3;
case 'S3S3'
filter_mask = S3' * S3;
case 'E3L3'
filter_mask = E3' * L3;
case 'L3E3'
filter_mask = L3' * E3;
case 'S3L3'
filter_mask = S3' * L3;
case 'L3S3'
filter_mask = L3' * S3;
case 'S3E3'
filter_mask = S3' * E3;
case 'E3S3'
filter_mask = E3' * S3;
otherwise
error('wrong filter type given');
end
elseif filter_type(2)=='5',
L5 = [1 4 6 4 1]; E5 = [-1 -2 0 2 1]; S5 = [-1 0 2 0 -1];
W5 = [-1 2 0 -2 1]; R5 = [1 -4 6 -4 1]; L5L5mask = L5' * L5;
switch filter_type
case 'L5L5'
filter_mask = L5' * L5;
case 'E5E5'
filter_mask = E5' * E5;
case 'S5S5'
filter_mask = S5' * S5;
case 'W5W5'
filter_mask = W5' * W5;
case 'R5R5'
filter_mask = R5' * R5;
case 'L5E5'
filter_mask = L5' * E5;
case 'E5L5'
filter_mask = E5' * L5;

case 'L5S5'
filter_mask = L5' * S5;
case 'S5L5'
filter_mask = S5' * L5;
case 'L5W5'
filter_mask = L5' * W5;
case 'W5L5'
filter_mask = W5' * L5;
case 'L5R5'
filter_mask = L5' * R5;
case 'R5L5'
filter_mask = R5' * L5;
case 'E5S5'
filter_mask = E5' * S5;
case 'S5E5'
filter_mask = S5' * E5;
case 'E5W5'
filter_mask = E5' * W5;
case 'W5E5'
filter_mask = W5' * E5;
case 'E5R5'
filter_mask = E5' * R5;
case 'R5E5'
filter_mask = R5' * E5;
case 'S5W5'
filter_mask = S5' * W5;
case 'W5S5'
filter_mask = W5' * S5;
case 'S5R5'
filter_mask = S5' * R5;
case 'R5S5'
filter_mask = R5' * S5;
case 'W5R5'
filter_mask = W5' * R5;
case 'R5W5'
filter_mask = R5' * W5;
otherwise
error('wrong filter type given');
end
else
error('wrong filter type given');
end
image =file_name;
% % STEP 1 mask convolution

disp('Mask convolution -> in progress...');
tic
image_conv = conv2(image,filter_mask,'valid');
disp('Mask convolution -> done.');
% % STEP 2 statistic computation
av_filter = fspecial('average', [window_size window_size]);
switch statistic_type
case 'MEAN'
disp('Statistic computation (mean) -> in progress...');
image_conv_TEM = conv2(image_conv,av_filter,'valid');
disp('Statistic computation (mean) -> done.');
case 'ABSM'
disp('Statistic computation (abs mean) -> in progress...');
image_conv_TEM = conv2(abs(image_conv),av_filter,'valid');
disp('Statistic computation (abs mean) -> done.');
case 'STDD'
disp('Statistic computation (st deviation) -> in progress...');
image_conv_TEM = image_stat_stand_dev(image_conv, window_size);
disp('Statistic computation (st deviation) -> done...');
otherwise
error('wrong statistic type given');
end
switch normalization_type
case 'MINMAX'
image_out = normalize(image_conv_TEM);
case 'FORCON'
if filter_type(2)=='3'
image_conv_norm = conv2(image,L3L3mask,'valid');
else
image_conv_norm = conv2(image,L5L5mask,'valid');
end
switch statistic_type
case 'MEAN'
image_norm = conv2(image_conv_norm,av_filter,'valid');
case 'ABSM'
image_norm = conv2(abs(image_conv_norm),av_filter,'valid');
case 'STDD'
image_norm = image_stat_stand_dev(image_conv_norm, window_size);
end
image_out = normalize(image_conv_TEM ./ image_norm);
end
t = toc
% % present results, normalize before plotting
% figure,
% imshow(image_out); title(['filter type: ' filter_type ' statistic type: ' statistic_type ' norm type: ' normalization_type ' elapsed:' num2str(t)]);
% % %%%%%% FUNCTIONS USED %%%%%%%
function output_image = normalize(input_image)
MIN = min(min(input_image)); MAX = max(max(input_image));
output_image = (input_image - MIN) / (MAX - MIN);
end
function stand_dev = image_stat_stand_dev(im_in, window_size)
[im_height im_width] = size(im_in);
i = 1;
for row = (window_size+1)/2 : im_height-(window_size-1)/2,
j = 1;
for column = (window_size+1)/2 : im_width-((window_size-1)/2),

    shift = (window_size-1)/2;
buffer = im_in(row-shift : row+shift, column-shift : column+shift);
stand_dev(i, j) = std(buffer(1:end));
j = j + 1;
end
i = i + 1;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end