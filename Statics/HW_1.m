load Data.csv
[m,n] = size(Data);
data_1 = reshape(Data,[1,m*n]);
figure (1);
%��������ֲ�ͼ
cdfplot(data_1);
figure (2);
%����Ƶ�ʷֲ�ֱ��ͼ
histogram(data_1,'Normalization','probability');
figure (3);
%��������ͼ
boxplot(Data');