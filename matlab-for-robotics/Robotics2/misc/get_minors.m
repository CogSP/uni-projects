function minorDet = get_minors(matrix, k,dets)
    [rows, cols] = size(matrix);

    if rows == k && cols == k
        dets = [dets, det(matrix)]; % Aggiungo il determinante della matrice attuale
    elseif rows ~= k
        for i = 1:rows
            submatrix = [matrix(1:i-1, :); matrix(i+1:rows, :)];
            dets = get_minors(submatrix, k, dets); % Aggiungo il risultato ricorsivo alla variabile dets
        end
    else
        for i = 1:cols
            submatrix = [matrix(:, 1:i-1), matrix(:, i+1:cols)];
            dets = get_minors(submatrix, k, dets); % Aggiungo il risultato ricorsivo alla variabile dets
        end
    end
    minorDet = simplify(unique(dets)); % Restituisco la lista di determinanti
end
