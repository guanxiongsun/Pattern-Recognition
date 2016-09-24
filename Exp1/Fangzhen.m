Data = [0,0;1,0;0,1;1,1;2,1;1,2;2,2;3,2;6,6;7,6;8,6;7,7;8,7;9,7;7,8;8,8;9,8;8,9;9,9];
X = Data(:,1);
Y = Data(:,2);
scatter(X, Y, 'o');
k=2;
[centers , data_new] = train_classifier (Data, k);
scatter(X(data_new(:,3)==1),Y(data_new(:,3)==1),'*');
hold on
scatter(X(data_new(:,3)==2),Y(data_new(:,3)==2));