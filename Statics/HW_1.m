load Data.csv
[m,n] = size(Data);
data_1 = reshape(Data,[1,m*n]);
figure (1);
%画出经验分布图
cdfplot(data_1);
figure (2);
%画出频率分布直方图
histogram(data_1,'Normalization','probability');
figure (3);
%画出箱型图
boxplot(Data');