data_gamma = [];
for i = 1:24
    data = [];
    for j = ((i-1)*100+2401):(i*100+2400)
        data_img = imread(eval(['''','F:\EEG地形图\EEG地形图\gamma\被试2\',num2str(j),'.png','''']));
        data_imgS = reshape(data_img,1,420,435,3);
        eval(['data = ','[data;','data_imgS','];']);
        
    end
    eval(['data',num2str(i),'=data;']);
    eval(['data_gamma = ','[data_gamma;','data',num2str(i),'];']);
end
