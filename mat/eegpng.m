%data3 = reshape(data,2400,40,16);
%data4 = data3(:,1:32,1);
%data5 = data4(199,:);
%data = a1(1:100,:,2);
data = data_deljx(:,1:32,1);
%data1 = data(1510,:);
%data2 = rand(32,1); 
for i = 10127:12000
    fig = figure;
    
    topoplot(data(i,:),chanloc,'electrodes','off');
    
    frame = getframe(fig); % 获取frame
    
    RGB = frame2im(frame); % 将frame变换成imwrite函数可以识别的格式
     %eval(['load(','''','C:\Users\41721\Desktop\数据\DEAPmat\','s',num2str(m),'.mat','''',')'])
    png = imcrop(RGB,[145,35,434,419]);
    eval(['imwrite(','png',',','''',num2str(i),'.png','''',')']);
    
    %imwrite(img,'a.png'); % 保存到工作目录下，名字为"a.png"
    
    close;
end

