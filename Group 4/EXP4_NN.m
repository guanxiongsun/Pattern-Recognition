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

clear Testsamples1;
clear TestLabels1;

Testsamples1 = Train_val;
TestLabels1 = Train_val_labels;


[m, n] = size(TrainSamples);
c = 10;
b = m;
TrainSamples=mapminmax(TrainSamples);
Testsamples1=mapminmax(Testsamples1);
Trainlabels_new = -ones(m,c);
for i =1:m
    Trainlabels_new(i,Trainlabels(i)+1) = 1;
end

Data_batch = TrainSamples;
Label_batch = Trainlabels_new;

clear TrainSamples
clear Trainlabels_new

Data_batch = Data_batch';
Label_batch = Label_batch';
net=newff(Data_batch,Label_batch,[60]);

net.trainFcn = 'traingdx';
net.trainParam.epochs=100000;
net.trainParam.goal=0.000004;
net.trainParam.max_fail=50;
%net.trainParam.lr = 0.5;

net.layers{1}.transferFcn = 'tansig';
%net.layers{2}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';
%net.layers{3}.transferFcn = 'purelin';

%����ѵ��
net=train(net,Data_batch,Label_batch);



m = size(Testsamples1,1);



%��������
an=sim(net,Testsamples1');
result = zeros(m,1);
for i = 1:m
    result(i)= find(an(:,i)==max(an(:,i)))-1;
end
accuracy_test = sum(result == TestLabels1)/m
an=sim(net,Data_batch);
result = zeros(size(Data_batch,2),1);
for i = 1:b
    result(i)= find(an(:,i)==max(an(:,i)))-1;
end
accuracy_train = sum(result == Trainlabels)/size(Data_batch,2)