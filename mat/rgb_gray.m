%转化为灰度图
for i = 121:600
    fig = figure;
    data = eval(['imread(','''','F:\EEG地形图\EEG地形图\delta\被试1\',num2str(i),'.png','''',')']);
    gdata = rgb2gray(data);
    imshow(gdata);
    frame = getframe(fig);
    GRAY = frame2im(frame); 
    eval(['imwrite(','GRAY',',','''',num2str(i),'.png','''',')']);
    close;
end