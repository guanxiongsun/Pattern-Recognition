%ѵ�������֪��%
function [a] = TrainBinaryPerception(Data,Labels)
    [m,n]= size(Data);
    %��ʼ��A%
    a = rand(n,1);
    
    %*********�����ݽ��й淶��ʹ�����е�A*y��ֵ��>0***********%
    Data(Labels==2,:) = -Data(Labels==2,:);
    
    %�ݶ��½�����a%
    k=0;
    while 1
    %����׼����J��A��%
    temp_score = Data*a;
    if(temp_score(k+1)<=0)
        a = a+Data(k+1,:)';
    end
    if((Data*a)>0)
        break
    else
        k=mod(k+1,m);  
    end
    end
end