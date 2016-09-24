function [y] = GMMcluster(X, A_m, Mu_m, Sigma_m, K)
%X数据
%A_m A的矩阵 2维
%Mu_m Mu的矩阵 3维
%Sigma_m Sigma的矩阵 4维
%K 分类个数
%返回值y是一个m*1的向量，代表m个数据的分类结果


    [m,n] = size(X);

    Prob_gmm = zeros(m,K);        %Prob矩阵存储每个data对于每个gmm得到的概率值

    for i=1:K
    	A = A_m(:,i);
        Mu = Mu_m(:,:,i);
        Sigma = Sigma_m(:,:,:,i);
        Prob_gmm(:,i) = CalculateGMMvalue(X, A, Mu, Sigma);       %计算单个gmm的值的函数
    end
    
    y = zeros(m,1);
    for i=1:m
        temp = find(Prob_gmm(i,:)==max(Prob_gmm(i,:)));
        if (size(temp,2)>1)
            temp2 = temp(1,1);
            y(i) = temp2;
        else
            y(i) = temp;
        end
    end
end