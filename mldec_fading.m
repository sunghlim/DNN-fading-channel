% ML decoding function for fading channel

function [Dec, idx]= mldec_fading(y,C, h)
L = size(C,2);
if(L~=length(y))
    error('Codewords need to be the same length as y!');
end

fading=repmat(h,size(C,1),1);
[~, idx] = min(sum((fading.*(2*C-1)-repmat(y,size(C,1),1)).^2,2));

Dec = C(idx,:);

end