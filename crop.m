

%InputImage = input('Please enter the name of the image and its extension \n','s');
for i = 41:50
        InputImage = rgb2gray(imread(strcat('D:\images\face_1\',InputImage)));
end