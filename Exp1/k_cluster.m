load ClusterSamples
% mat = reshape(ClusterSamples(3,:),[28,28]);
% imagesc(mat);  
dataset = double(ClusterSamples);
y = double(SampleLabels);
[m, n] = size(dataset);
k =10;
[centers , data_new] = train_classifier (dataset, k);
imshow()
%T用来计数，是一个10*10的矩阵，记录每个聚类中不同数字的个数%
T = zeros(10,10);
for i=1:10
   kclster = y(data_new(:,n+1)==i);
   d = size(kclster,1);
   for j = 1:d
       T(i, kclster(j)+1)=T(i, kclster(j)+1)+1;
   end
end
disp(T)