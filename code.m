%Author: Upama Nakarmi
%Date: 5/7/2016

clear all;
close all;
clc;

%Number of images = 50
M= 50;

%read and show the images.
S=[];   %img matrix

for i = 1 : M
    str  =  strcat(int2str(i),'.jpg');   %concatenates two strings that form the name of the image
    eval('img=rgb2gray(imread(str));'); %read the image and convert it to grayscale or 2D image
    
    %figure(1);
    %subplot(5,10,i);
    %imshow(img);
    
    [irow icol]  =  size(img);    % get the number of rows (irow) and columns (icol)
    temp         =  reshape(img,1,irow*icol);     %creates a 1x36000 matrix for this dataset
    S            =  [S temp'];                      %S contains 50 images each of 1X3600
                        
end



%take the mean of the images
mean_image = double(mean(S,2));
size(mean_image);


%images without the mean in them
img_not_mean = [];
for j = 1 : M
    
    temp1        =  double((S(:,j)))- mean_image;
    img_not_mean = [img_not_mean temp1];
    
end


%finding the covariance matrix
covariance = img_not_mean' * img_not_mean;
size(covariance);


%finding the eigen vectors
[eigen_vectors eigen_values] = pcacov(covariance);
size(eigen_vectors);
size(eigen_values);
eigen_values;
eigen_vectors;


%only taking the first 40 eigen vectors
v=[];
d=[];
for k = 1:10
    v = [v eigen_vectors(:,k)];
end

size(v);
size(img_not_mean);
d;


%convert eigen vectors of lower dimension to a vector of higher
%dimension which is an eigen face so, we have 40 eigen faces.
eigen_face = [];
for l = 1:10
    temp2 = (img_not_mean) * (v(:,l));
    eigen_face = [eigen_face temp2];
end
%imshow(reshape(eigen_face(:,1),200,180));
size(eigen_face);


%finding 40 weights of every orignal 50 faces.
%Hence, there will be 200 weights in total, 40 for each face.
weight = [];

for m  = 1:50
    for n = 1:10
        temp3 = ((img_not_mean(:,m))') * eigen_face(:,n);
        weight = vertcat(weight, temp3);
    end
end

weight = reshape(weight,10,50);
size(weight(:,1));
size(eigen_face(:,1)');
%weight(1,1);

s = (double(weight(1,1)) * double((eigen_face(:,1)))')';
size(s);

%reconstruct the original image from the old eigen vectors and the weights.
face  = [];
temp4 = [];
total_sum   = [];

for o = 1:50
   for p = 1:10
        temp4 = [temp4 (double(weight(p,o)) * double((eigen_face(:,p))'))'];
        size(temp4);
        
   end
        size(temp4); 
   
        total_sum = sum(temp4,2);
        size(total_sum);
        %figure(2);
        %subplot(5,10,o);
        %imshow(reshape(total_sum,200,180));
   
        face  = [face total_sum];
        temp4 = [];
        total_sum = [];
end


InputImage = input('Please enter the name of the image and its extension \n','s');
InputImage = rgb2gray(imread(strcat('D:\images\face_1\',InputImage)));
%subplot(1,2,1);
%figure(3);
%imshow(InputImage); colormap('gray');title('Test image','fontsize',18);

%reshape the new image
new_image = double(reshape(InputImage,1,36000));

%subtract the image from the mean image
new_image_not_mean = new_image - mean_image';

%find the weights of the new image
weight_new = [];
for z = 1:10
    temp_new = new_image_not_mean * eigen_face(:,z);
    weight_new = [weight_new temp_new];
end

new_weight = weight_new';
size(weight_new);

%calculate distance of the new image from the old images
distance  = [];
temp_dist = 0;
for y = 1: 50
    for x = 1:10
            temp_dist = ((new_weight(x,1)) - weight(x,y))^2 + temp_dist; 
    end
    distance  = [distance temp_dist];
    temp_dist = 0;
end

distance;

if((max(distance) - min(distance))< 2.14e+17)
    input('Face not recognized');
else
    input('Face recognized');
end
size(distance);

minimum = min(distance);


















    