%% Task 3 – Intra-database analysis: DeepFake detection usando transfer learning con AlexNet

% 1. Preparación del entorno y carga de datos
clear; close all; clc;

% Definir rutas de la base de datos
devFolder = fullfile('../data/Task3/development_cropped'); % Carpeta de desarrollo
evalFolder = fullfile('../data/Task3/evaluation_cropped'); % Carpeta de evaluación

% Crear un imageDatastore para el conjunto de desarrollo (training)
imdsTrain = imageDatastore(devFolder, ...
'IncludeSubfolders', true, ...
'LabelSource', 'foldernames');

% Verificar la distribución de etiquetas
countEachLabel(imdsTrain)

%% 2. Definir la nueva arquitectura basada en la imagen proporcionada
numClasses = 2;
layers = [
    imageInputLayer([124 124 3], 'Name', 'input')

    % Primera capa convolucional + Max Pooling
    convolution2dLayer(7, 32, 'Padding', 'same', 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(3, 'Stride', 2, 'Name', 'maxpool1')

    % Segunda capa convolucional + Max Pooling
    convolution2dLayer(5, 64, 'Padding', 'same', 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(3, 'Stride', 2, 'Name', 'maxpool2')

    % Tercera capa convolucional + Max Pooling
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(3, 'Stride', 2, 'Name', 'maxpool3')

    % Flatten
    fullyConnectedLayer(512, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc1')

    % Capa densa con dropout
    fullyConnectedLayer(128, 'Name', 'fc2')
    reluLayer('Name', 'relu_fc2')
    dropoutLayer(0.5, 'Name', 'dropout')

    % Capa de salida
    fullyConnectedLayer(numClasses, 'Name', 'fc_output')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classOutput')
];

% Crear el gráfico de capas
lgraph = layerGraph(layers);

%% 3. Configuración de data augmentation y preprocesamiento
% Redimensionar las imágenes al tamaño de entrada de AlexNet
inputSize = [124 124 3]; % Tamaño de la imagen de entrada
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);

%% 4. Configuración de las opciones de entrenamiento
options = trainingOptions('sgdm', ...
'MiniBatchSize', 300, ...
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

% Clasificamos
predictedLabels = classify(trainedNet, augimdsEval);
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