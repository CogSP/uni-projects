Nbits = 8 %% in the exercise this was the number of bits of the encoder for measuring a single turn
xgray=[0 1 1 0 1 0 0 1] % from Most Significant Bit to Least Significant Bit

% Gray to binary
xbin(1)=xgray(1);
for i=1:Nbits-1
	xbin(i+1)=xor(xbin(i),xgray(i+1));
end

xbin

% binary to decimal
xdec=xbin(Nbits);
for i=1:Nbits-1
	xdec=xdec+xbin(Nbits-i)*2^i;
end

% then xdec can be multiplied by the resolution in order to obtain the angle. This because xdec indicates the number of "spicchi" 
% that the angle covers, and since each "spicchio" is a resolution long in terms of degrees/radians, xdec * res gives the 
% whole angle covered, in degrees/radians

xdec

