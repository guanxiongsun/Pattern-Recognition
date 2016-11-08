clear
load TestLabels1.csv
load Testsamples1.csv
load TrainSamples.csv
load Trainlabels.csv

Data = [TrainSamples Trainlabels];

%����Data��һ��20000x121�ľ��󷵻�18000,2000���Ӽ�
[Train_t,Train_t_labels, Train_val, Train_val_labels] = DivideData(Data, 0.8);
clear TrainSamples
clear Trainlabels
TrainSamples = Train_t;
Trainlabels = Train_t_labels;

[m, n] = size(TrainSamples);
c = 10;
%��ʼ��w
W = rand(n,c)*0.0001;
%�������ݼ���w
TrainSamples = [TrainSamples ones(m,1)];
W = [W;zeros(1,c)];
W_min = [W;zeros(1,c)];
%����Loss,�涨lamda
%����mini-batch�ݶ��½����� b= 256
lamda = 50000;
b = 256;
lr = 1e-7;
iter = 0;
max_iter = 1000;
threshold = 0.9;
loss_min = 100;
while true
    %�����ݶ�ʹ��~С�����ݶ��½���
    %�����������batch
    selecter = randi([1,m],b,1);
    Data_batch = TrainSamples(selecter,:);
    Label_batch = Trainlabels(selecter);
    [Loss, Delta] = Calculate_loss(Data_batch, Label_batch, W, lamda);
    if(Loss < loss_min)
       loss_min = Loss;
       W_min =  W;
    end
    W = W - lr*Delta;
    if (Loss < threshold || iter > max_iter)
        break;
    else
        iter
        Loss
        iter = iter + 1;
    end
end
clear Testsamples1;
clear TestLabels1;

Testsamples1 = Train_val;
TestLabels1 = Train_val_labels;

m = size(Testsamples1,1);
Testsamples1 = [Testsamples1 ones(m,1)];
Test_score = Testsamples1*W;
Result = zeros(m,1);
for i=1:m
    Result(i) = find(Test_score(i,:)==max(Test_score(i,:)))-1;
end
temp = (Result==TestLabels1);
accuracy = sum(temp)/m;
fprintf('The accuracy rate is: %f',accuracy);