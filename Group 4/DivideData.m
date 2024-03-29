function [Train_t,Train_t_labels, Train_val, Train_val_labels] = DivideData(Data, percent)
    [m,n] = size(Data);
    selector = randperm(m);
    sel_t = selector(1:percent*m)';
    sel_val = selector(percent*m+1:m)';
    Data_t = Data(sel_t,:);
    Data_val = Data(sel_val,:);
    Train_t = Data_t(:,1:n-1);
    Train_t_labels = Data_t(:,n);
    Train_val = Data_val(:,1:n-1);
    Train_val_labels = Data_val(:,n);
end
