load Samples

%处理数据增加偏置列%
Data = [ones(size(TrainSamples,1),1),TrainSamples];
K = 10;
[A] = TrainPerception(Data,Labels, K);