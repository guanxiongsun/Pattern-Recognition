%����������Ҫ��Data�Ǵ���ƫ���еģ�����Labelsֻ�����1��2%
function [a] = TrainBinaryLMSE(Data,Labels)
    [m,~] = size(Data);
    b = ones(m,1);
    %�����ݽ��б�׼��%
    Data_new = Data;
    Data_new(Labels==2,:) = -Data(Labels==2,:);
    
    Y= Data_new;
    Y_p = (Y'*Y)\Y';
    a = Y_p*b;
end