%% Task 2

% 1. Preparación del entorno y carga de datos
clear; close all; clc;

% Definir rutas de la base de datos
evalFolder = fullfile('../data/Task2/cropped_faces'); % Carpeta de evaluación

% Cargar el modelo previamente entrenado
load('../models/AlexNet_task2.mat', 'trainedNet');

%% Evaluación en el conjunto de evaluación
% Cargar el conjunto de evaluación
% Definir rutas de la base de datos
fakeFolder = fullfile(evalFolder, 'fake');
realFolder = fullfile(evalFolder, 'real');

% Crear imageDatastore para cada categoría
imdsFake = imageDatastore(fakeFolder, 'IncludeSubfolders', true);
imdsReal = imageDatastore(realFolder, 'IncludeSubfolders', true);

% Asignar etiquetas manualmente asegurando el mismo tipo de datos
imdsFake.Labels = categorical(repmat("fake", numel(imdsFake.Files), 1));
imdsReal.Labels = categorical(repmat("real", numel(imdsReal.Files), 1));

% Unir ambos imageDatastores
imdsEval = imageDatastore(cat(1, imdsFake.Files, imdsReal.Files), ...
    'Labels', cat(1, imdsFake.Labels, imdsReal.Labels));

% Comprobar que las etiquetas son correctas
disp(countEachLabel(imdsEval));

% Asegurar que el tamaño de las imágenes coincida con el modelo
inputSize = trainedNet.Layers(1).InputSize;
augimdsEval = augmentedImageDatastore(inputSize(1:2), imdsEval);

% Obtener etiquetas correctas desde las subcarpetas
% Asumimos que las subcarpetas dentro de evalFolder se llaman "real" y "fake"
fileLabelsEval = imdsEval.Labels; % MATLAB ya asignó las etiquetas basadas en las carpetas

% Convertir etiquetas a 'categorical' asegurando el orden correcto
imdsEval.Labels = categorical(fileLabelsEval, {'fake', 'real'}); 

% Clasificar las imágenes de evaluación
[YPredEval, scoresEval] = classify(trainedNet, augimdsEval);
YTrueEval = imdsEval.Labels;

% Calcular la exactitud
accEval = mean(YPredEval == YTrueEval);
fprintf('Exactitud en evaluación: %.2f%%\n', accEval * 100);

% Preparar datos para la curva ROC
positiveClass = categorical({'fake'}); % Definir la clase positiva
YTrueBinEval = double(YTrueEval == positiveClass);

% Obtener índice de la clase positiva en las predicciones
classes = categories(YTrueEval);
posIdx = find(classes == positiveClass);
scoresPosEval = scoresEval(:, posIdx);

% Calcular y graficar la curva ROC
[XEval, YEval, TEval, AUCEval] = perfcurve(YTrueBinEval, scoresPosEval, 1);
figure;
plot(XEval, YEval, 'LineWidth', 2);
xlabel('Tasa de Falsos Positivos');
ylabel('Tasa de Verdaderos Positivos');
title(['Curva ROC - Evaluación, AUC = ' num2str(AUCEval)]);
grid on;