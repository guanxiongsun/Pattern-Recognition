load ClusterSamples
% mat = reshape(ClusterSamples(3,:),[28,28]);
% imagesc(mat);  
dataset = double(ClusterSamples);
y = double(SampleLabels);
[m, n] = size(dataset);
k =10;
[centers , data_new] = train_classifier (dataset, k);
imshow()
%T������������һ��10*10�ľ��󣬼�¼ÿ�������в�ͬ���ֵĸ���%
T = zeros(10,10);
for i=1:10
   kclster = y(data_new(:,n+1)==i);
   d = size(kclster,1);
   for j = 1:d
       T(i, kclster(j)+1)=T(i, kclster(j)+1)+1;
   end
end
disp(T)