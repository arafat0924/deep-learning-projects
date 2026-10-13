function delta = bc(w, z, delta_next, df)
%BC Manual chain-rule backpropagation for one fully connected layer.
%   w is the next layer's weight matrix and df is the current layer's
%   activation derivative evaluated at its pre-activation z.

delta = df(z) .* (w' * delta_next);
end
