function [y] = GMMcluster(X, A_m, Mu_m, Sigma_m, K)
%X����
%A_m A�ľ��� 2ά
%Mu_m Mu�ľ��� 3ά
%Sigma_m Sigma�ľ��� 4ά
%K �������
%����ֵy��һ��m*1������������m�����ݵķ�����


    [m,n] = size(X);

    Prob_gmm = zeros(m,K);        %Prob����洢ÿ��data����ÿ��gmm�õ��ĸ���ֵ

    for i=1:K
    	A = A_m(:,i);
        Mu = Mu_m(:,:,i);
        Sigma = Sigma_m(:,:,:,i);
        Prob_gmm(:,i) = CalculateGMMvalue(X, A, Mu, Sigma);       %���㵥��gmm��ֵ�ĺ���
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