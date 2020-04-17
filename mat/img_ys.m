%转化为灰度图
data_gamma3_32 = [];
% 将2400分24次读取
for i = 1:24
    data_s = [];
    % 每次读取100张图
    for j = ((i-1)*100+4801):(i*100+4800)
        % 读取地形图数据（mat）
        data = eval(['imread(','''','F:\EEG地形图\EEG地形图\gamma\被试3\',num2str(j),'.png','''',')']);
        % 缩小地形图数据的尺寸
        data_rs = imresize(data,[32,32]);
        %imshow(data_rs);
        % RBG转化为灰度图
        gdata = rgb2gray(data_rs);
        data_re = reshape(gdata,1,32,32);
        %imshow(data_re);
        %eval(['imwrite(','png',',','''',num2str(j),'.png','''',')']);
        % 将100张图的灰度矩阵拼接成data_s
        eval(['data_s = ','[data_s;','data_re','];']);
        close;
    end
    % 将24个data_s拼接构成data_alpha1_32的数据，即第一个被试的alpha波的32*32灰度矩阵数据
    eval(['data_y',num2str(i),'=data_s;']);
    eval(['data_gamma3_32 = ','[data_gamma3_32;','data_y',num2str(i),'];']);
    
end