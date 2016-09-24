function [y] = CalculateGMMvalue(X, A, Mu, Sigma)
%计算单个GMM的值  参数已知
    m = size(X,1);
    k= size(A,1);
    
    Prob = zeros(m,k);
    
    for i=1:k
        Prob(:,i) = mvnpdf(X,Mu(i,:),Sigma(:,:,i)); 
    end
    y = Prob*A;
end