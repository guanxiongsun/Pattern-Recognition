load Samples

%������������ƫ����%
Data = [ones(size(TrainSamples,1),1),TrainSamples];
K = 10;
[A] = TrainPerception(Data,Labels, K);