load TrainSamples.csv
load TrainLabels.csv
load TestLabels.csv
load TestSamples.csv

K = 10;
k = 1;

while 1
%构造训练样本矩阵 三维  第三维代表类别 m*n*K

[m,n] = size(TrainSamples);
Train = cell(K,1);
%每个训练样本的大小并不相同  所以不能用三维数组 改用cell矩阵存储  每个cell是一个矩阵

for i=1:K
    Train{i} = TrainSamples(TrainLabels==(i-1),:);
end


%构造参数空矩阵  
%A_m   k*K k是高斯数，K是类别数
%Mu_m  k*n*K
%Sigma_m  n*n*k*K

A_m = zeros(k,K);
Mu_m = zeros(k,n,K);
Sigma_m = zeros(n,n,k,K);


%循环训练得到参数矩阵
for i=1:K
    [A_m(:,i), Mu_m(:,:,i), Sigma_m(:,:,:,i)]= learnGMM(Train{i},k);
    i
end

% [A1, Mu1, Sigma1] = learnGMM(Train1, 2);
% [A2, Mu2, Sigma2] = learnGMM(Train2, 2);
%disp(A1);
%disp(Mu1);
%disp(Sigma1);
%disp(A2);
%disp(Mu2);
%disp(Sigma2);
% K=2;
% A_m(:,1) = A1;
% Mu_m(:,:,1) = Mu1;
% Sigma_m(:,:,:,1) = Sigma1;
% A_m(:,2) = A2;
% Mu_m(:,:,2) = Mu2;
% Sigma_m(:,:,:,2) = Sigma2;
y = GMMcluster(TestSamples,A_m,Mu_m,Sigma_m,K);
y = y-1;
%disp(y);

fprintf('The number of Guasses is : %d \n', k);
fprintf('The accuracy is : %f \n', sum((y-TestLabels)==0)/10000);

k=k+1;
if k>5
    break
end
end