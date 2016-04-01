function [Result] = MA_Deviation(tsObj,MADays)

%/ calculate MA Deviation

%/ Result: Row -> MA , Column -> asset

    Result = zeros(size(MADays,2),size(tsObj,2));

    for i = 1:size(MADays,2)
        TS = fts2mat(tsObj); %/ export data
        MA = mean(TS(end-MADays(i)+1:end,:));   %/find MA   
        Result(i,:) = MA./TS(end,:); %/ calculate divation from MA
    end

end
