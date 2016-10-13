load SimpleSamples.mat

%训练感知器分类器，输入数据DATA, Labels, k是种类数%
%处理训练集增加一列偏置1%

SimpleTest_new = [ones(size(SimpleTest,1),1),SimpleTest];
w1 = SimpleTest(SimpleLabels==1,:);
w2 = SimpleTest(SimpleLabels==2,:);
figure(1);
plot(w1(:,1),w1(:,2),'r+')
hold on
plot(w2(:,1),w2(:,2),'*')
[a] = TrainBinaryPerception(SimpleTest_new,SimpleLabels);
x=linspace(0,2);
y=-a(2)*x/a(3)-a(1)/a(3);
plot(x,y);
figure(2);
[a_2] = TrainBinaryLMSE(SimpleTest_new,SimpleLabels);
plot(w1(:,1),w1(:,2),'r+')
hold on
plot(w2(:,1),w2(:,2),'*')
x=linspace(0,2);
y=-a_2(2)*x/a_2(3)-a_2(1)/a_2(3);
plot(x,y);