function [centers , data_new] = train_classifier (dataset, k)
    [m, n] = size(dataset);
    in_c0 = randi([1, m], [k,1],'distributed');
    in_c1 = (1:k)';
    in_c2 = [1;2;3;4;5;7;9;13;18;23];
    %初始化聚类中心%
    centers_n = dataset(in_c0,:);
    centers = zeros(size(centers_n));
    data_new = [dataset zeros(m, 1)];
    %循环直到中心不变%
    ttt=1;
    while (sum(sum(abs(centers_n - centers)))>0.1)
    %标记数据集%
    dist = zeros(k,1);
    for i=1:m
        for j =1:k
            dist(j) = norm(dataset(i, :)-centers_n(j,:));
        end
        temp = find(dist==min(dist));
        data_new(i, n+1) = temp;
    end    
    centers=centers_n;
    %重新计算中心%
    for i=1:k
        centers_n(i,:) = mean(dataset(data_new(:,n+1) == i, :) ,1);
    end
    ttt=ttt+1
    sum(sum(abs(centers_n - centers)))
    end
    centers = centers_n;
end