%% Task 1 – Intra-database analysis: DeepFake detection usando transfer learning con AlexNet

% 1. Preparación del entorno y carga de datos
clear; close all; clc;

% Definir rutas de la base de datos
devFolder = fullfile('../data/Task1/development_cropped'); % Carpeta de desarrollo
evalFolder = fullfile('../data/Task1/evaluation_cropped'); % Carpeta de evaluación

% Crear un imageDatastore para el conjunto de desarrollo (training)
imdsTrain = imageDatastore(devFolder, ...
'IncludeSubfolders', true, ...
'LabelSource', 'foldernames');

% Verificar la distribución de etiquetas
countEachLabel(imdsTrain)

%% 2. Cargar la red preentrenada AlexNet y modificarla para 2 clases
net = alexnet;
inputSize = net.Layers(1).InputSize;

% Convertir la red a un layerGraph (útil para modificar capas)
if isa(net, 'SeriesNetwork')
lgraph = layerGraph(net.Layers);
else
lgraph = layerGraph(net);
end

% Número de clases (se espera, por ejemplo, 'Fake' y 'Real')
numClasses = numel(categories(imdsTrain.Labels));

% Reemplazar las últimas capas para adaptar a la clasificación binaria
newLayers = [
fullyConnectedLayer(numClasses, 'Name', 'fc8_new')
softmaxLayer('Name', 'softmax_new')
classificationLayer('Name', 'classOutput_new')];

lgraph = replaceLayer(lgraph, 'fc8', newLayers(1));
lgraph = replaceLayer(lgraph, 'prob', newLayers(2));
lgraph = replaceLayer(lgraph, 'output', newLayers(3));

%% 3. Configuración de data augmentation y preprocesamiento
% Redimensionar las imágenes al tamaño de entrada de AlexNet
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);

%% 4. Configuración de las opciones de entrenamiento
options = trainingOptions('sgdm', ...
'MiniBatchSize', 32, ...
'MaxEpochs', 25, ... % Ajustar según la cantidad de datos y tiempo disponible
'InitialLearnRate', 1e-4, ...
'Shuffle', 'every-epoch', ...
'Verbose', true, ...
'Plots','training-progress');

%% 5. Entrenar la red
trainedNet = trainNetwork(augimdsTrain, lgraph, options);

%% 6. Evaluación en el conjunto de desarrollo (Training evaluation)
% Clasificar las imágenes del conjunto de desarrollo
[YPredTrain, scoresTrain] = classify(trainedNet, augimdsTrain);
YTrueTrain = imdsTrain.Labels;

% Calcular la exactitud
accTrain = mean(YPredTrain == YTrueTrain);
fprintf('Exactitud en desarrollo: %.2f%%\n', accTrain*100);

% Para la curva ROC, se asume que la clase positiva es "Fake"
categoriesList = categories(YTrueTrain);
positiveClass = categoriesList(contains(string(categoriesList), "fake"));

[~, posIdx] = ismember(positiveClass, categoriesList);

% Convertir las etiquetas verdaderas a binario: 1 para "Fake", 0 para "Real"
YTrueBinTrain = double(YTrueTrain == positiveClass);
scoresPosTrain = scoresTrain(:, posIdx);

% Calcular la curva ROC y AUC utilizando perfcurve
[XTrain, YTrain, TTrain, AUCTrain] = perfcurve(YTrueBinTrain, scoresPosTrain, 1);

% Graficar la ROC del conjunto de desarrollo
figure;
plot(XTrain, YTrain, 'LineWidth', 2);
xlabel('Tasa de Falsos Positivos');
ylabel('Tasa de Verdaderos Positivos');
title(['Curva ROC - Desarrollo, AUC = ' num2str(AUCTrain)]);
grid on;

%% 7. Evaluación en el conjunto de evaluación (final evaluation)
% Cargar el conjunto de evaluación
imdsEval = imageDatastore(evalFolder, ...
'IncludeSubfolders', true, ...
'LabelSource', 'foldernames');
augimdsEval = augmentedImageDatastore(inputSize(1:2), imdsEval);

% Clasificar las imágenes de evaluación
[YPredEval, scoresEval] = classify(trainedNet, augimdsEval);
YTrueEval = imdsEval.Labels;

% Calcular la exactitud
accEval = mean(YPredEval == YTrueEval);
fprintf('Exactitud en evaluación: %.2f%%\n', accEval*100);

% Preparar datos para la ROC en evaluación
YTrueBinEval = double(YTrueEval == positiveClass);
scoresPosEval = scoresEval(:, posIdx);
[XEval, YEval, TEval, AUCEval] = perfcurve(YTrueBinEval, scoresPosEval, 1);

% Graficar la ROC del conjunto de evaluación
figure;
plot(XEval, YEval, 'LineWidth', 2);
xlabel('Tasa de Falsos Positivos');
ylabel('Tasa de Verdaderos Positivos');
title(['Curva ROC - Evaluación, AUC = ' num2str(AUCEval)]);
grid on;

