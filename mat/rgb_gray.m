%ת��Ϊ�Ҷ�ͼ
for i = 121:600
    fig = figure;
    data = eval(['imread(','''','F:\EEG����ͼ\EEG����ͼ\delta\����1\',num2str(i),'.png','''',')']);
    gdata = rgb2gray(data);
    imshow(gdata);
    frame = getframe(fig);
    GRAY = frame2im(frame); 
    eval(['imwrite(','GRAY',',','''',num2str(i),'.png','''',')']);
    close;
end