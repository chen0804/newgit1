%ת��Ϊ�Ҷ�ͼ
data_gamma3_32 = [];
% ��2400��24�ζ�ȡ
for i = 1:24
    data_s = [];
    % ÿ�ζ�ȡ100��ͼ
    for j = ((i-1)*100+4801):(i*100+4800)
        % ��ȡ����ͼ���ݣ�mat��
        data = eval(['imread(','''','F:\EEG����ͼ\EEG����ͼ\gamma\����3\',num2str(j),'.png','''',')']);
        % ��С����ͼ���ݵĳߴ�
        data_rs = imresize(data,[32,32]);
        %imshow(data_rs);
        % RBGת��Ϊ�Ҷ�ͼ
        gdata = rgb2gray(data_rs);
        data_re = reshape(gdata,1,32,32);
        %imshow(data_re);
        %eval(['imwrite(','png',',','''',num2str(j),'.png','''',')']);
        % ��100��ͼ�ĻҶȾ���ƴ�ӳ�data_s
        eval(['data_s = ','[data_s;','data_re','];']);
        close;
    end
    % ��24��data_sƴ�ӹ���data_alpha1_32�����ݣ�����һ�����Ե�alpha����32*32�ҶȾ�������
    eval(['data_y',num2str(i),'=data_s;']);
    eval(['data_gamma3_32 = ','[data_gamma3_32;','data_y',num2str(i),'];']);
    
end