% Crear detector de rostros basado en Haar
faceDetector = vision.CascadeObjectDetector();

% Carpeta con imágenes de evaluación
evalFolder = fullfile('evaluation'); % Ajusta la ruta según corresponda
%evalFolder = fullfile('development'); % si es para development acordarse
%de descomentar las 4 lineas AÑADIDAS de abajo
imdsEval = imageDatastore(evalFolder, 'IncludeSubfolders', true);

% Carpeta de salida (donde se guardarán las imágenes recortadas)
outputFolder = fullfile('cropped_faces');
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Procesar cada imagen
for i = 1:numel(imdsEval.Files)
    % Leer imagen
    img = imread(imdsEval.Files{i});
    
    % Obtener la etiqueta de la imagen a partir de la estructura de carpetas
    [filePath, name, ext] = fileparts(imdsEval.Files{i});
    [parentFolder, subFolder] = fileparts(filePath); % Obtiene "persona_X" como subFolder
    [~, category] = fileparts(parentFolder); % Obtiene "fake" o "real"

    % Crear carpeta de categoría en outputFolder (cropped_faces/fake o cropped_faces/real)
    categoryFolder = fullfile(outputFolder, category);
    if ~exist(categoryFolder, 'dir')
        mkdir(categoryFolder);
    end

    % Guardar la imagen en la carpeta adecuada
    % Detectar rostros en la imagen
    bboxes = step(faceDetector, img);
    
    if ~isempty(bboxes)
        % Seleccionar el rostro más grande
        [~, idx] = max(bboxes(:, 3) .* bboxes(:, 4));
        faceROI = bboxes(idx, :);
        
        % Recortar la cara detectada
        croppedFace = imcrop(img, faceROI);
        
        % Guardar la imagen recortada en la carpeta correcta (fake o real)
        outputPath = fullfile(categoryFolder, [name '_face' ext]);
        imwrite(croppedFace, outputPath);
        %outputPath_fliped = fullfile(categoryFolder, [name '_facef' ext]); %% AÑADIDA%%
        %imwrite(flip(croppedFace,2), outputPath_fliped);                   %% AÑADIDA%%
    else
        % Si no se detecta cara, guardar la imagen original en la misma carpeta
        outputPath = fullfile(categoryFolder, [name '_noface' ext]);
        imwrite(img, outputPath);
        %outputPath_fliped = fullfile(categoryFolder, [name '_nofacef' ext]); %% AÑADIDA%%
        %imwrite(flip(croppedFace,2), outputPath_fliped);                     %% AÑADIDA%%
    end
end

disp('Proceso completado. Las imágenes recortadas están en:');
disp(outputFolder);
