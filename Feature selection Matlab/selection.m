% veri=table2array(train1);


pt = cvpartition(train1.Cover_Type,"HoldOut",0.3);

veriTrain = veri(training(pt),:);
veriTest = veri(test(pt),:);

veriTrain_X=veriTrain(:,1:55);
veriTrain_Y=veriTrain(:,56);
veriTest_X=veriTest(:,1:55);
veriTest_Y=veriTest(:,56);

%dataFitcecoc = fitcecoc(veriTrain_X,veriTrain_Y);
dataTree = fitctree(veriTrain_X,veriTrain_Y);
dataKnn= fitcknn(veriTrain_X,veriTrain_Y);
%%
[labelDT,scoreDT,costDT] = predict(dataTree,veriTest_X);
[mDT,orderDT]=confusionmat(labelDT,veriTest_Y);

figure
cmDT = confusionchart(labelDT,veriTest_Y);
cmDT.ColumnSummary = 'column-normalized';
cmDT.RowSummary = 'row-normalized';
cmDT.Title = 'DT Confusion Matrix';
%%
% [labelSVM,scoreSVM,costSVM] = predict(dataFitcecoc,veriTest_X);
% [mSVM,orderSVM]=confusionmat(labelSVM,veriTest_Y);
% 
% figure
% cmSVM = confusionchart(labelSVM,veriTest_Y);
% cmSVM.ColumnSummary = 'column-normalized';
% cmSVM.RowSummary = 'row-normalized';
% cmSVM.Title = 'SVM Confusion Matrix';

%%
[labelKNN,scoreKNN,costKNN] = predict(dataKnn,veriTest_X);
[mKNN,orderKNN]=confusionmat(labelKNN,veriTest_Y);

figure
cmKNN = confusionchart(labelKNN,veriTest_Y);
cmKNN.ColumnSummary = 'column-normalized';
cmKNN.RowSummary = 'row-normalized';
cmKNN.Title = 'KNN Confusion Matrix';
%%
% mSVM_Result_common = multiclass_metrics_common(mSVM);
% [mSVMsonuc,mSVMreference] = multiclass_metrics_special(mSVM);
%%
mDT_Result_common = multiclass_metrics_common(mDT);
[mDTsonuc,mDTreference] = multiclass_metrics_special(mDT);
%%
mKNN_Result_common = multiclass_metrics_common(mKNN);
[mKNNsonuc,mKNNreference] = multiclass_metrics_special(mKNN);
%%

% X=veri(:,1:2);
% 
% veriTable=table2array(veri);
figure
gscatter(veriTest_X(:,10),veriTest_X(:,5),veriTest_Y);
h=gca;
lims = [h.XLim h.YLim]; % Extract the x and y axis limits
title('{\bf Dağılım Şeması}');
xlabel('date');
ylabel('shop__id');
legend('FontSize',14,'Location','northeastoutside');