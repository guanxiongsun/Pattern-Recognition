clear
load TestLabels1.csv
load Testsamples1.csv
load TrainSamples.csv
load Trainlabels.csv
[m, n] = size(TrainSamples);
c = 10;
b = 20000;
% ��ʼ��w
% W = rand(n,c)*0.0001;
TrainSamples=mapminmax(TrainSamples);
Testsamples1=mapminmax(Testsamples1);
Trainlabels_new = -ones(m,c);
for i =1:m
    Trainlabels_new(i,Trainlabels(i)+1) = 1;
end

selecter = randi([1,m],b,1);
%Data_batch = TrainSamples;
%Label_batch = Trainlabels_new;
Data_batch = TrainSamples(selecter,:);
Label_batch = Trainlabels_new(selecter,:);

clear TrainSamples
clear Trainlabels_new

Data_batch = Data_batch';
Label_batch = Label_batch';
net=newff(Data_batch,Label_batch,84);

net.trainFcn = 'traingdx';
net.trainParam.epochs=100000;
net.trainParam.goal=0.000004;
net.trainParam.max_fail=15;
%net.trainParam.lr = 0.01;

net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';
%net.layers{3}.transferFcn = 'purelin';
%net.layers{3}.transferFcn = 'purelin';

%����ѵ��
net=train(net,Data_batch,Label_batch);
an=sim(net,Testsamples1');
result = zeros(m,1);
for i = 1:m
    result(i)= find(an(:,i)==max(an(:,i)))-1;
end
accuracy_test = sum(result == TestLabels1)/m
an=sim(net,Data_batch);
result = zeros(m,1);
for i = 1:b
    result(i)= find(an(:,i)==max(an(:,i)))-1;
end
accuracy_train = sum(result == Trainlabels(selecter))/m