load Train1.csv
load Train2.csv
load Test1.csv
load Test2.csv
[A1, Mu1, Sigma1] = learnGMM(Train1, 2);
[A2, Mu2, Sigma2] = learnGMM(Train2, 2);
%disp(A1);
%disp(Mu1);
%disp(Sigma1);
%disp(A2);
%disp(Mu2);
%disp(Sigma2);
K=2;
A_m(:,1) = A1;
Mu_m(:,:,1) = Mu1;
Sigma_m(:,:,:,1) = Sigma1;
A_m(:,2) = A2;
Mu_m(:,:,2) = Mu2;
Sigma_m(:,:,:,2) = Sigma2;
y = GMMcluster(Test2,A_m,Mu_m,Sigma_m,K);
disp(y);
sum(y==2)

