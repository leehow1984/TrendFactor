function [Result] = TrendFactor_PositionControl(MarketData,MADays,SmoothingFactor,reb_freq)

%/ Market Data: Financial Time Series Object
%/ MADays: Moving average day vector 1 X N
%/ SmoothingFactor: Moving Average beta smoothing vector 1 X N
%/ Rebalancing frequency: 'D' = Daily, 'W' = Weekly, 'M' = Monthly, 'Q'=
%/ Quaterly

%/ convert market data into the new rebalance frequency
NewfreqFTS = convertto(MarketData,reb_freq);
Dates = NewfreqFTS.dates;
Dates = Dates(SmoothingFactor+1:end,:);
Names = fieldnames(MarketData);
Names = transpose(Names(4:end,1));
b_vec = zeros(0);
stats_vec= zeros(0);

    for i = 2:size(Dates,1)
        l_tm1 = find(MarketData.dates == Dates(i-1));      
        tsObj_tm1 = MarketData(l_tm1-max(MADays)+1:l_tm1);
        MADev_tm1 = MA_Deviation(tsObj_tm1,MADays);
        l2_tm1 = find(NewfreqFTS.dates == Dates(i-1)); 
        RetMat_t =  fts2mat(NewfreqFTS(l2_tm1+1))./ fts2mat(NewfreqFTS(l2_tm1)) -1;
        
        %/ b: beta coefficient
        %/ bint: lower and upper confidence bound
        %/ r:residual
        %/ rint:
        %/ stats: Stats: R^2, Fstats,Pvalue
        
        x = transpose(RetMat_t);
        y = [ones(size(transpose(MADev_tm1),1),1) transpose(MADev_tm1)];
        [b,bint,r,rint,stats] =  regress(x,y);
        b_tran = transpose(b);
        %/ beta coefficient vector
        if stats(1,3) > 0.05 
           b_tran(:,:) = nan;
        end    
        b_vec = [b_vec; b_tran];
        stats_vec = [stats_vec; (stats)];
    end
    
    ExpRetRank_vec = zeros(0);
    RetMat_tplus1_vec = zeros(0);
    ExpRet_vec = zeros(0);
    quantile = 5;
    
    %/ predict return
    for i = SmoothingFactor+1:size(Dates,1)-1
        %/calculate MA deviation at t 
        l_t = find(MarketData.dates == Dates(i-1)); 
        tsObj_t = MarketData(l_t-max(MADays)+1:l_t);
        MADev_t = MA_Deviation(tsObj_t,MADays); 
        %/ find smoothed beta coefficient
        %/ b_smooth = mean(b_vec(i-SmoothingFactor:i-1,:));
        %/ find expected return
        %/ ExpRet = b_vec(i-SmoothingFactor:i-1,2:end) * MADev_t + b_vec(i-SmoothingFactor:i-1,1);
        bx = b_vec(i-SmoothingFactor:i-1,2:end) * MADev_t;
        a = b_vec(i-SmoothingFactor:i-1,1);
        bx(isnan(bx(:,1)),:)=[];
        a(isnan(a(:,1)),:)=[];
        ExpRet = mean(bsxfun(@plus,bx,a),1);
        
        %/ ExpRet = b_vec(i-1,2:end) * MADev_t + b_vec(i-1,1);
        %/ Ranking
        ExpRetRank = transpose(ceil(quantile * tiedrank(ExpRet) / length(ExpRet)));
        
        %/ actual future return
        l2_t = find(NewfreqFTS.dates == Dates(i)); 
        RetMat_tplus1 =  fts2mat(NewfreqFTS(l2_t))./ fts2mat(NewfreqFTS(l2_t-1)) -1;
        
        %/ export reslut
        ExpRet_vec = [ExpRet_vec; ExpRet];
        ExpRetRank_vec = [ExpRetRank_vec;transpose(ExpRetRank)];
        RetMat_tplus1_vec = [RetMat_tplus1;RetMat_tplus1_vec];
    end
    
    %/performance analysis
    AverageRet = zeros(1,quantile);
    PandL_sub =  ones(1,quantile);
    PandL_tot = ones(1,1);
    LPosition_vec = cell(1,20);
    SPosition_vec = cell(1,20);
    for i = 1:size(ExpRetRank_vec,1)
        for j = 1:quantile
            AverageRet(1,j) = AverageRet(1,j) + mean(RetMat_tplus1_vec(i,ExpRetRank_vec(i,:)==j));
            PandL_sub(i+1,j) = PandL_sub(i,j) * (1 + mean(RetMat_tplus1_vec(i,ExpRetRank_vec(i,:)==j)));   
        end
            if i == 1 
               LPosition_vec(i,1:size(Names(1,ExpRetRank_vec(i,:)==quantile),2)) = Names(1,ExpRetRank_vec(i,:)==quantile);
               SPosition_vec(i,1:size(Names(1,ExpRetRank_vec(i,:)==1),2)) = Names(1,ExpRetRank_vec(i,:)==1);
            else
               LPosition_vec(size(LPosition_vec,1)+1,1:size(Names(1,ExpRetRank_vec(i,:)==quantile),2)) = Names(1,ExpRetRank_vec(i,:)==quantile);
               SPosition_vec(size(SPosition_vec,1)+1,1:size(Names(1,ExpRetRank_vec(i,:)==1),2)) = Names(1,ExpRetRank_vec(i,:)==1);            
            end     
           PandL_tot(i+1,1) = PandL_tot(i,1) * ( 1 + mean(RetMat_tplus1_vec(i,ExpRetRank_vec(i,:) == quantile)) - mean(RetMat_tplus1_vec(i,ExpRetRank_vec(i,:)==1)) ) ;
    end 
    
    plot(Dates(SmoothingFactor+1:end,1),PandL_tot)
    datetick('x','yy/mmm')
    
    %/Positioning analysis
    Unique_Ccy = cell(0,1);
    for i = 1:size(Names,2)
        ccy = Names{i};
        Unique_Ccy(size(Unique_Ccy,1)+1,1) = {ccy(1:3)};
        Unique_Ccy(size(Unique_Ccy,1)+1,1) = {ccy(4:6)};
    end 
    Unique_Ccy = transpose(unique(Unique_Ccy));
    %/ sum up positioning
    Exposure_Ccy = zeros(1,size(Unique_Ccy,2));
    for i = 1:size(LPosition_vec,1)
        LLongCcy = cellfun(@(c)c(1:3),LPosition_vec(i,(~cellfun(@isempty,LPosition_vec(i,:)))),'UniformOutput',false);
        LShortCcy = cellfun(@(c)c(4:6),LPosition_vec(i,(~cellfun(@isempty,LPosition_vec(i,:)))),'UniformOutput',false);
        SLongCcy = cellfun(@(c)c(4:6),SPosition_vec(i,(~cellfun(@isempty,SPosition_vec(i,:)))),'UniformOutput',false);
        SShortCcy = cellfun(@(c)c(1:3),SPosition_vec(i,(~cellfun(@isempty,SPosition_vec(i,:)))),'UniformOutput',false);        
        Exposure_Ccy(i,:) = 0;
        for j = 1:size(Unique_Ccy,2)
            Exposure_Ccy(i,j) = Exposure_Ccy(i,j) + sum(strcmp(LLongCcy,Unique_Ccy(1,j)))/sum(cellfun(@isempty,LPosition_vec(i,:)));
            Exposure_Ccy(i,j) = Exposure_Ccy(i,j)-sum(strcmp(LShortCcy,Unique_Ccy(1,j)))/sum(cellfun(@isempty,LPosition_vec(i,:)));
            Exposure_Ccy(i,j) = Exposure_Ccy(i,j)+sum(strcmp(SLongCcy,Unique_Ccy(1,j)))/sum(cellfun(@isempty,SPosition_vec(i,:)));
            Exposure_Ccy(i,j) = Exposure_Ccy(i,j)-sum(strcmp(SShortCcy,Unique_Ccy(1,j)))/sum(cellfun(@isempty,SPosition_vec(i,:)));
        end
        
    end
    
end 

function [TopPCWeight,BottomPCWeight] = PositionControlFunction(TopPosition,BottomPosition,MaxWeight)
   





end