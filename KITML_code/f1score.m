function [Fmicro, Fmacro,accm] = f1score(pred, act)

% per(i,1) false negative rate
%           = (false negatives)/(all output negatives)
% per(i,2) false positive rate
%           = (false positives)/(all output positives)
% per(i,3) true positive rate or sensitivity or recall
%           = (true positives)/(all output positives)
% per(i,4) true negative rate or SPECIFICITY 
%           = (true negatives)/(all output negatives)
% precision = TP/(TP+FP)
% Recall =TP/(TP+FN)
n=length(unique(act));
Mact = sparse(act,1:length(act),ones(length(act),1),n,length(act));
Mact=full(Mact);


Mpred = sparse(pred,1:length(pred),ones(length(pred),1),n,length(pred));
Mpred=full(Mpred);

[c,cm,ind,per] = confusion(Mact,Mpred);

%TP is there, just FP and FN
TP=diag(cm);
FP=sum(cm)'-TP;
FN=sum(cm,2)-TP;

Pmicro=sum(TP)/sum(TP+FP);
Rmicro=sum(TP)/sum(TP+FN);
Fmicro=2*Pmicro*Rmicro/(Pmicro+Rmicro);

if(sum(TP)~=n)
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!');
    tprecision=0;
    trecall=0; 
    for k=1:n
        if(TP(k)==0)
            if(FP(k)==0 && FN(k)==0)
                tprecision=tprecision+1;
                trecall=trecall+1;
            elseif(FP(k)==0 && FN(k)~=0)
                 tprecision=tprecision+1;
                 trecall=trecall+0;
            elseif(FP(k)~=0 && FN(k)==0)
                 tprecision=tprecision+0;
                 trecall=trecall+1;
            elseif(FP(k)~=0 && FN(k)~=0)
                 tprecision=tprecision+0;
                 trecall=trecall+0;
            end
        else
            tprecision=tprecision+TP(k)/(TP(k)+FP(k));
            trecall=trecall+TP(k)/(TP(k)+FN(k));
        end      
    end
    Pmacro=tprecision/n;
    Rmacro=trecall/n;
else
    Pmacro=sum(TP./(TP+FP))/n;
    Rmacro=sum(TP./(TP+FN))/n;
end
Fmacro=2*Pmacro*Rmacro/(Pmacro+Rmacro);
accm=1-c;