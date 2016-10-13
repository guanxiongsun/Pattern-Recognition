%分类器输入要求Data是带有偏置列的，而且Labels只标记有1和2%
function [a] = TrainBinaryLMSE(Data,Labels)
    [m,~] = size(Data);
    b = ones(m,1);
    %将数据进行标准化%
    Data_new = Data;
    Data_new(Labels==2,:) = -Data(Labels==2,:);
    
    Y= Data_new;
    Y_p = (Y'*Y)\Y';
    a = Y_p*b;
end