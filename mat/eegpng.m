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
    
    frame = getframe(fig); % ��ȡframe
    
    RGB = frame2im(frame); % ��frame�任��imwrite��������ʶ��ĸ�ʽ
     %eval(['load(','''','C:\Users\41721\Desktop\����\DEAPmat\','s',num2str(m),'.mat','''',')'])
    png = imcrop(RGB,[145,35,434,419]);
    eval(['imwrite(','png',',','''',num2str(i),'.png','''',')']);
    
    %imwrite(img,'a.png'); % ���浽����Ŀ¼�£�����Ϊ"a.png"
    
    close;
end

