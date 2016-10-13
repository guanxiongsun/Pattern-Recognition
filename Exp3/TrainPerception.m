%通过data和Labels训练感知器，种类数为k
function [A] = TrainPerception(Data,Labels, K)
%DATA,LABELS,k是种类数%
    [~,n] = size(Data);
    A = zeros(n,K);
    %使用一对多的方式训练多类分类器%
    for i=1:K
        %将数据分成两类，第i类和非第i类%
        Labels_new = 2*ones(size(Labels));
        Labels_new(Labels == i-1) = 1;
        A(:,i) = TrainBinaryPerception(Data,Labels_new);
    end
end
    