function [A, Mu, Sigma] = learnGMM(Data, k)    %Data is the training data; k is the number of Gauss in GMM%
    [m, n] = size(Data);

    threshold = 0.0000001;

   
    A =  randi([1,5*k],k, 1);                  %A matric is k*1  %                       
    A = A/sum(sum(A));                         %sum of A =1%
    A_n = zeros(size(A));
    
    temp = randi([1,m],k, 1);
    Mu = Data(temp,:);                         %Mu is k*n select in Data%
    Mu_n = zeros(size(Mu));    
    
    Sigma = zeros(n,n,k);
    Sigma_n = zeros(n,n,k);
    for i=1:k                 
        Sigma(:,:,i) = eye(n)*1000000;             %Sigma is n*n*k的三维矩阵第三维是分量k%
    end
    for i=1:k
        Sigma_n(:,:,i) = zeros(n,n);
    end
    
    Y_p = zeros(m, k);                                             %y的概率矩阵m*k%
    Prob = zeros(m, k);
    
    
    %ttt=0;
    while 1
    %计算P(t,k)
    
    for i=1:k
        Prob(:,i) = mvnpdf(Data,Mu(i,:),Sigma(:,:,i));                 %Prob是Data对于不同分量的概率矩阵m*k%
    end
    
    for t=1:m
        for i=1:k
            Y_p(t,i) = (A(i)*Prob(t,i))/(Prob(t,:)*A);
        end
    end
    
    %重新计算参数
    
    for i=1:k
        A_n(i) =  mean(Y_p(:,i));
    end
    
    temp1 = zeros(m,n);
    for i=1:k
        for t=1:m
            temp1(t,:)=Data(t,:)*(Y_p(t,i));            
        end
        Mu_n(i,:) = sum(temp1)/sum(Y_p(:,i));
    end
    
    for i=1:k
        temp2 = zeros(n,n);
        for t=1:m
            temp2 = temp2+((((Data(t,:)-Mu_n(i,:))')*(Data(t,:)-Mu_n(i,:)))*Y_p(t,i));
        end
        Sigma_n(:,:,i) = temp2/sum(Y_p(:,i));
    end
    
    if(sum(sum(abs(Mu_n - Mu)))+sum(sum(sum(abs(Sigma_n-Sigma))))<threshold)
    %if(norm(Mu_n - Mu)==0&&norm(Sigma_n(:,:,1)-Sigma(:,:,1))==0&&norm(Sigma_n(:,:,2)-Sigma(:,:,2))==0)
    %if(abs(norm(Mu_n - Mu))+abs(norm(Sigma_n(:,:,1)-Sigma(:,:,1)))+abs(norm(Sigma_n(:,:,2)-Sigma(:,:,2)))<0.000001)
        break
    else
        
        %abs(norm(Mu_n - Mu))+abs(norm(Sigma_n(:,:,1)-Sigma(:,:,1)))+abs(norm(Sigma_n(:,:,2)-Sigma(:,:,2)))
        
        Mu = Mu_n;
        A = A_n;
        Sigma = Sigma_n;
        %ttt =ttt+1
    end
    end