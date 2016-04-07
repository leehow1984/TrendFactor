function [Result] = TrendFactor_TimeSeries(MarketData,MADays,LookBackFactor,reb_freq)

%/ Market Data: Financial Time Series Object
%/ MADays: Moving average day vector 1 X N
%/ SmoothingFactor: Moving Average beta smoothing vector 1 X N
%/ Rebalancing frequency: 'D' = Daily, 'W' = Weekly, 'M' = Monthly, 'Q'=
%/ Quaterly

%/ convert market data into the new rebalance frequency
%/ can only run weekly strategy at the moment
NewfreqFTS = convertto(MarketData(max(MADays):end),reb_freq);
Dates = NewfreqFTS.dates;
%/Dates = Dates(LookBackFactor:end,:);
FtsInfo = ftsinfo(MarketData);


MADev_vec = zeros(0);
RetMat_vec = zeros(0);
b_vec = zeros(0);
stats_vec= zeros(0);
Exp_Ret_vec= zeros(size(Dates,1)-LookBackFactor-1,size(FtsInfo.seriesnames,1));

    % calculate time series of MA Deviation and Return 
 
    for i = 1:size(FtsInfo.seriesnames,1)
        %/ get time series data for each security
        Security_tsObj = extfield(MarketData,FtsInfo.seriesnames{i,1});
        Security_Deviation = MA_Deviation_TimeSeries(Security_tsObj,MADays);
        %/ find deviation for all testing dates
        Security_Deviation = Security_Deviation(ismember(MarketData.dates,Dates,'rows'),:);
        
        %/ calculate t + 1 return
        Security_tsObj_freq = Security_tsObj(ismember(Security_tsObj.dates,Dates,'rows'));
        RetMat_t =  fts2mat(Security_tsObj_freq(2:end))./ fts2mat(Security_tsObj_freq(1:end-1)) -1;
        
        %/ run regression
        for j = LookBackFactor:size(RetMat_t,1)-1
            x = [ones(LookBackFactor,1) Security_Deviation(j - LookBackFactor + 1:j,:)];
            y = RetMat_t(j - LookBackFactor + 1:j,1);
            [b,bint,r,rint,stats] =  regress(y,x);
            Exp_Ret = b(1,1) + Security_Deviation(j + 1,:) * b(2:end,1);
            Exp_Ret_vec(j - LookBackFactor + 1,i)= Exp_Ret;
        end
        
        RetMat_vec = [RetMat_vec RetMat_t];
        
    end 
 
    ExpRetRank_vec = zeros(0);
    RetMat_tplus1_vec = zeros(0);
    quantile = 10;
    AverageRet = zeros(1,quantile);
    PandL_sub =  ones(1,quantile);
    PandL_tot = ones(1,1);    
    
    %/ predict return
    for i = 1:size(Exp_Ret_vec,1)

        %/ rank return
        ExpRetRank = transpose(ceil(quantile * tiedrank(Exp_Ret_vec(i,:)) / length(Exp_Ret_vec(i,:))));
        ExpRetRank_vec = [ExpRetRank_vec;transpose(ExpRetRank)];
        RetMat_tplus1 = RetMat_vec(LookBackFactor+i,:);
        
        %/ calculate average return and sub return
        for j = 1:quantile 
            AverageRet(1,j) = AverageRet(1,j) + mean(RetMat_tplus1(ExpRetRank_vec(i,:)==j));
            PandL_sub(i+1,j) = PandL_sub(i,j) * (1 + mean(RetMat_tplus1(ExpRetRank_vec(i,:)==j)));   
        end
        
        PandL_tot(i+1,1) = PandL_tot(i,1) * ( 1 + mean(RetMat_tplus1(ExpRetRank_vec(i,:) == quantile)) - mean(RetMat_tplus1(ExpRetRank_vec(i,:)==1)) ) ;
        
    end
    
result= 0;    
    



