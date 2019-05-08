% ML decoding function

function [Dec, idx]= mldec(y,C)
L = size(C,2);
if(L~=length(y))
    error('Codewords need to be the same length as y!');
end

[~, idx] = min(sum(((2*C-1)-repmat(y,size(C,1),1)).^2,2));

Dec = C(idx,:);

end