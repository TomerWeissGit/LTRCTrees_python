from __future__ import annotations
from collections.abc import Iterable
import numpy as np
import pandas as pd
from typing import Optional
from lifelines import CoxPHFitter
from sklearn.tree import DecisionTreeRegressor
from sksurv.tree import SurvivalTree


#' Fit a relative risk survival tree for LTRC data
#'
#' \code{LTRCART} returns an \link[rpart]{rpart} object. This function extends
#' the survival tree algorithm in \code{\link{rpart}} to fit left-truncated and
#' right censored (LTRC) data.
#'
#' @param formula A formula object specifies the regression function, with the
#' response be a \link[survival]{Surv} object, with form Surv(time1, time2, event)
#' @param data An optional data frame which contains the variables
#' named in the formula.
#' @param weights Optional case weights, same as in \code{\link{rpart}}
#' @param subset Optional expression saying that only a subset of the rows of the data should be
#' used in the fit, same as in \code{\link{rpart}}
#' @param no.SE Number of standard errors used in pruning, with default value 0.
#' @param control A list of control values used to control the \code{\link{rpart}}
#' algorithm, with default cp = 0.001. See \link[rpart]{rpart.control} for details.
#' @return An object of class rpart. See \link[rpart]{rpart.object}.
#'
#' @references Fu, W. and Simonoff, J.S. (2017). Survival trees for left-truncated and right-censored data,
#' with application to time-varying covariate data. Biostatistics 18 (2), 352-369.
#'
#'
#' @examples
#' ## The Assay of serum free light chain data in survival package
#' ## Adjust data & clean data
#' library(survival)
#' library(LTRCtrees)
#' Data <- flchain
#' Data <- Data[!is.na(Data$creatinine),]
#' Data$End <- Data$age + Data$futime/365
#' DATA <- Data[Data$End > Data$age,]
#' names(DATA)[6] <- "FLC"
#'
#' ## Setup training set and test set
#' Train = DATA[1:500,]
#' Test = DATA[1000:1020,]
#'
#' ## Fit LTRCART survival tree
#' ## make sure to attach survival package (by library(survival) ) before using Surv function
#' LTRCART.obj <- LTRCART(Surv(age, End, death) ~ sex + FLC + creatinine, Train)
#'
#' ## Putting Surv(End, death) in formula would result an error message
#' ## since LTRCART is expecting Surv(time1, time2, event)
#'
#' ## Plot the fitted tree
#' library(rpart.plot)
#' rpart.plot(LTRCART.obj)
#'
#' ## Plot as partykit::party object
#' library(partykit)
#' plot(as.party(LTRCART.obj))
#'
#' ## Plot as partykit::party object with survival curves on terminal nodes
#' LTRCART.obj.party <- as.party(LTRCART.obj)
#' LTRCART.obj.party$fitted[["(response)"]]<- Surv(Train$age, Train$End, Train$death)
#' plot(LTRCART.obj.party)
#'
#' ## Predict relative risk on test set
#' LTRCART.pred <- predict(LTRCART.obj, newdata = Test)
#'
#'
#'
#'####################################################################
#'####### Survival tree with time-varying covariates ##################
#'####################################################################
#'## The pbcseq dataset of survival package
#'library(survival)
#'## Create the start-stop-event triplet needed for coxph and LTRC trees
#'first <- with(pbcseq, c(TRUE, diff(id) !=0)) #first id for each subject
#'last <- c(first[-1], TRUE) #last id
#'time1 <- with(pbcseq, ifelse(first, 0, day))
#'time2 <- with(pbcseq, ifelse(last, futime, c(day[-1], 0)))
#'event <- with(pbcseq, ifelse(last, status, 0))
#'event <- 1*(event==2)
#'
#'pbcseq$time1 <- time1
#'pbcseq$time2 <- time2
#'pbcseq$event <-  event
#' ## Fit the Cox model and LTRCART tree with time-varying covariates
#'fit.cox <- coxph(Surv(time1, time2, event) ~ age + sex + log(bili), pbcseq)
#'LTRCART.fit <- LTRCART(Surv(time1, time2, event) ~ age + sex + log(bili), pbcseq)
#'rpart.plot(LTRCART.fit)
#'
#'### transform the wide format data into long format data using tmerge function
#'### from survival function
#'## Stanford Heart Transplant data
#'jasa$subject <- 1:nrow(jasa)
#'
#'tdata <- with(jasa, data.frame(subject = subject,
#'                               futime= pmax(.5, fu.date - accept.dt),
#'                               txtime= ifelse(tx.date== fu.date,
#'                                              (tx.date -accept.dt) -.5,
#'                                              (tx.date - accept.dt)),
#'                               fustat = fustat))
#'
#'sdata <- tmerge(jasa, tdata, id=subject,death = event(futime, fustat),
#'                  trt = tdc(txtime), options= list(idname="subject"))
#'
#'sdata$age <- sdata$age - 48
#'
#'sdata$year <- as.numeric(sdata$accept.dt - as.Date("1967-10-01"))/365.25
#'
#'Cox.fit <- coxph(Surv(tstart, tstop, death) ~ age+ surgery, data= sdata)
#'LTRCART.fit <- LTRCART(Surv(tstart, tstop, death) ~ age + transplant, data = sdata)
#'rpart.plot(LTRCART.fit)
#'
#'@export
from survival_data import SurvivalData

class LTRCart:
    def __init__(self,survival_data_obj: SurvivalData, x:pd.DataFrame, weights: Optional[Iterable] = None, no_se: Optional[float] = 0.0, control: Optional[dict] = None):
        self.data = survival_data_obj.survival_data.copy()
        self.weights = weights
        self.no_se = no_se
        self.control = control
        self.x =x

    def ltrc_art(self):
        y = self.data
        x = self.x
        status = self.data.event.copy()
        times = self.data.time2.copy()
        unique_death_times = times[status==1]
        cox_ph = CoxPHFitter()
        cox_baseline_cum_hazard = cox_ph.fit(self.data,'time2','event').baseline_cumulative_hazard_
        # remove inf hazard and Nones.
        cox_baseline_cum_hazard = (cox_baseline_cum_hazard[~cox_baseline_cum_hazard
                                    .isin([np.nan, np.inf, -np.inf]).any(1)])
        cox_baseline_hazard_death = cox_baseline_cum_hazard.loc[cox_baseline_cum_hazard.index.isin(unique_death_times)]
        cum_haz_times = [0]+list(cox_baseline_hazard_death.index)[:-1]+[times.max()]
        cum_haz = [0]+ list(cox_baseline_hazard_death.iloc[:,0])

        start_time_cum_hazard = np.interp(cum_haz_times, cum_haz, [0]+list(y.time1))
        end_time_cum_hazard = np.interp(cum_haz_times, cum_haz, [0]+list(y.time2))
        new_time = end_time_cum_hazard-start_time_cum_hazard
        new_surv_df = pd.DataFrame({'event':status,'time':new_time})
        tree_model =DecisionTreeRegressor(criterion="poisson",**self.control)
        tree_model.fit(X=self.x,y=new_surv_df,sample_weight=self.weights)

        Newtime < - End.cumhaz - Start.cumhaz
        Formula = formula(paste(c(paste("cbind(Newtime,", "Status)", sep=""), predictors), collapse="~"))
        tree_model < - rpart::rpart(formula=Formula, method="poisson", weights=weights, subset=subset, control=control)


LTRCART <- function(formula, data, weights=NULL, subset = NULL, no.SE = 0, control = rpart::rpart.control(cp = 0.001)){

  if(missing(data)){
    y <- eval(formula[[2]])
    predictors <- formula[[3]]

    if (!inherits(y, "Surv") | length(as.list(formula[[2]]))!=4){
      stop(" Response must be a 'survival' object with 'Surv(time1, time2, event)' ")
    }

    Status <- y[,3L]
    Times <- y[,2L]

    ##unique death times
    unique.times <- sort(unique(Times[Status == 1]))

    temp <- survival::coxph(y ~ 1)
    cumhaz.table <- survival::basehaz(temp)

    #Check if Inf hazard exists
    if(sum(is.infinite(cumhaz.table$hazard))!=0){
      cumhaz.table <- cumhaz.table[cumhaz.table$hazard != Inf,] ## subset(cumhaz.table, hazard != Inf)
    }

    #cumhaz.table2 <- subset(cumhaz.table, time %in% unique.times)
    cumhaz.table2 <- cumhaz.table[cumhaz.table$time %in% unique.times,]

    cumhaz.times <- c(0, cumhaz.table2$time[-length(cumhaz.table2$time)], max(Times))
    cumhaz <- c(0, cumhaz.table2$hazard)

    Start.cumhaz <- stats::approx(cumhaz.times, cumhaz, y[,1L])$y
    End.cumhaz <- stats::approx(cumhaz.times, cumhaz, y[,2L])$y

    Newtime <- End.cumhaz - Start.cumhaz
    Formula = formula(paste(c( paste("cbind(Newtime,","Status)",sep = ""), predictors), collapse = "~"))
    Fit.tree <- rpart::rpart(formula = Formula, method = "poisson", weights = weights, subset = subset, control = control)

    ##pruning the tree
    if( length(unique(Fit.tree$where))!=1 ){
      cventry <- which.min(Fit.tree$cptable[, "xerror"])

      if(no.SE == 0){
        cpcv <- Fit.tree$cptable[cventry, "CP"]
        Fit.tree <- rpart::prune(Fit.tree,cp=cpcv)
      }else{
        xerrorcv <- Fit.tree$cptable[cventry, "xerror"]
        sexerrorcv <- xerrorcv + Fit.tree$cptable[cventry,"xstd"] * no.SE
        cpcvse <- Fit.tree$cptable[which.max(Fit.tree$cptable[, "xerror"] <= sexerrorcv), "CP"]
        Fit.tree <- rpart::prune(Fit.tree, cp = cpcvse)
      }
    }
    return(Fit.tree)
  }else{
    Data <- data
    ###if in the form of Surv(time1,time2,event)~predictors with data following
    Response <- formula[[2]]
    predictors <- formula[[3]]

    if(length(as.list(Response))!=4){
      stop(" Response must be a 'survival' object with 'Surv(time1, time2, event)' ")
    }
    y.names <- c(as.character(as.list(Response)[[2]]),as.character(as.list(Response)[[3]]),as.character(as.list(Response)[[4]]))
    y.IDs <- match(y.names, names(Data))
    y <- survival::Surv(Data[,y.IDs[1]],Data[,y.IDs[2]],Data[,y.IDs[3]])

    Status <- y[,3L]
    Times <- y[,2L]

    ##unique death times
    unique.times <- sort(unique(Times[Status == 1]))
    temp <- survival::coxph(y ~ 1)
    cumhaz.table <- survival::basehaz(temp)

    #Check if Inf hazard exists
    if(sum(is.infinite(cumhaz.table$hazard))!=0){
      cumhaz.table <- cumhaz.table[cumhaz.table$hazard != Inf,] ## subset(cumhaz.table, hazard != Inf)
    }

    #cumhaz.table2 <- subset(cumhaz.table, time %in% unique.times)
    cumhaz.table2 <- cumhaz.table[cumhaz.table$time %in% unique.times,]

    cumhaz.times <- c(0, cumhaz.table2$time[-length(cumhaz.table2$time)], max(Times))
    cumhaz <- c(0, cumhaz.table2$hazard)

    Start.cumhaz <- stats::approx(cumhaz.times, cumhaz, y[,1L])$y
    End.cumhaz <- stats::approx(cumhaz.times, cumhaz, y[,2L])$y

    Data$Newtime <- End.cumhaz - Start.cumhaz
    Formula = formula(paste(c( paste("cbind(Newtime,",y.names[3],")",sep = ""), predictors), collapse = "~"))
    DATA = Data[,-y.IDs[1:2]]
    Fit.tree <- rpart::rpart(formula = Formula, data = DATA, method = "poisson", weights = weights, subset = subset, control = control)

    ##pruning the tree
    if( length(unique(Fit.tree$where))!=1 ){
      cventry <- which.min(Fit.tree$cptable[, "xerror"])

      if(no.SE == 0){
        cpcv <- Fit.tree$cptable[cventry, "CP"]
        Fit.tree <- rpart::prune(Fit.tree,cp=cpcv)
      }else{
        xerrorcv <- Fit.tree$cptable[cventry, "xerror"]
        sexerrorcv <- xerrorcv + Fit.tree$cptable[cventry,"xstd"] * no.SE
        cpcvse <- Fit.tree$cptable[which.max(Fit.tree$cptable[, "xerror"] <= sexerrorcv), "CP"]
        Fit.tree <- rpart::prune(Fit.tree, cp = cpcvse)
      }
    }
    return(Fit.tree)
  }
}
