rm(list=ls())
library('rstan')
library('ggplot2')
library('BayesFactor')

SubEx <- c('SPS902','SPS926')
LevNum <- 2
modelFile <- '~/Experiments/SPS/batch/model_two.stan'

nIter     <- 10000
nChains   <- 4
nWarmup   <- floor(nIter/2)
nThin     <- 1
rstan_options(auto_write = TRUE)
options(mc.cores = 4)

RootPath <- '~/Experiments/SPS'
OutPath1 <- file.path(RootPath,'results')
if (! dir.exists(OutPath1)) dir.create(OutPath1)

OutPath <- file.path(OutPath1,paste0(LevNum,'model'))
if (! dir.exists(OutPath)) dir.create(OutPath)

DataPath <- file.path(RootPath,'data')
Files <- dir(DataPath,'txt')
subinfo <- read.table(file.path(RootPath,'batch','SPS_subinfo.txt'),sep='\t',header=T)

TxtFile <- file.path(OutPath1,'SPS_prep_all_logit.txt')
if (file.exists(TxtFile)) {
  dat <- read.table(TxtFile,sep='\t',header=T)
  dat.original <- dat
}
if (!file.exists(TxtFile)) {
for (file in Files) {
  fsp <- strsplit(file,split='_')
  SubString <- strsplit(file,split='_')
  SubNum <- SubString[[1]][1]
  SubVer <- SubString[[1]][2]
  sn1 <- as.numeric(strsplit(SubNum,split='SPS')[[1]][2])
  if (floor(sn1/100)==9) grp <- 'alpha'
  if (floor(sn1/100)==8) grp <- 'sham'
  if (floor(sn1/100)==7) grp <- 'beta'
  guess1 <- subinfo$guess[subinfo$Sub==sn1]
  age1 <- subinfo$age[subinfo$Sub==sn1]
  sex1 <- subinfo$sex[subinfo$Sub==sn1]
  dat.t2 <- read.table(file.path(DataPath,file),sep='\t',header=T)
  
  dat.t21 <- subset(dat.t2,run!=0)
  # compute task accuracy
  dat.t22 <- dat.t21
  dat.t22$AgtAlt_des[which(dat.t22$AgtAlt_des==-99)] <- 0
  dat.t22$IsCorrect[which(dat.t22$IsCorrect==-99)] <- 0
  acc1 <- mean(dat.t22$AgtAlt_des)
  acc2 <- mean(subset(dat.t22,probe_if==1)$IsCorrect)
  if (sn1==935) {
    otime1 <- (dat.t21$SpkRat_onset - dat.t21$dummy_onset)/60
    otime2 <- (dat.t21$AdsRat_onset - dat.t21$dummy_onset)/60
    dat.t21 <- dat.t21[otime1<=13&otime2<=13,]
  }
  ll <- nrow(dat.t21)
  attach(dat.t21)
  dat.lm <- data.frame(SubName=rep(SubNum,ll),
                       Version=rep(SubVer,ll),
                       GrpNum=grp,item=item,
                       speech_act_no=speech_act_no,
                       run=run, AgtAlt_des=AgtAlt_des,
                       spk=SpkRat_scr,ads=AdsRat_scr,
                       ACC1=acc1,
                       ACC2=acc2,
                       age=age1,
                       guess=guess1,
                       sex=sex1
                       )
  detach(dat.t21)
  ifelse(file==Files[1],dat<-dat.lm,dat<-rbind(dat,dat.lm))
}

# define groups
dat$Grp <- rep(0,nrow(dat))
dat$Grp[which(dat$GrpNum=='alpha')] <- 1
dat$Grp[which(dat$GrpNum=='sham')] <- 2
dat$Grp[which(dat$GrpNum=='beta')] <- 3

dat$y <- rep(0,nrow(dat))
dat$y[which(dat$speech_act_no%in%c(3,4))] <- 1
dat$y[which(dat$speech_act_no%in%c(1,2))] <- 0

dat$guessc <- rep(0,nrow(dat))
dat$guessc[dat$guess==1] <- 1
dat$guessc[dat$guess==2] <- -1
dat$guessc[dat$guess==3] <- 0

str(dat)

dat.original <- dat
write.table(dat.original,TxtFile,sep='\t',row.names=F)
}

dat.acc <- aggregate(ACC1 ~ SubName*Grp*ACC2*age, FUN=mean, data=dat)
acc1_lob <- mean(dat.acc$ACC1)-2*sd(dat.acc$ACC1)
acc2_lob <- mean(dat.acc$ACC2)-2*sd(dat.acc$ACC2)

## remove wrong trial
dat <- subset(dat.original,! AgtAlt_des %in% c(0, -99))

dat <- subset(dat,! SubName %in% SubEx)

# acc test
dat.acc <- aggregate(ACC1 ~ SubName * Grp * ACC2 * age, FUN = mean, data = dat)
dat.acc$Grp <- as.factor(dat.acc$Grp)
str(dat.acc)
library(dplyr)
dat.acc %>% group_by(Grp) %>% summarize(mean(ACC1), sd(ACC1),
                                        mean(ACC2), sd(ACC2))
summary(t.acc1 <- aov(ACC1 ~ Grp, data = dat.acc))
1/anovaBF(ACC1 ~ Grp, data = dat.acc)
effectsize::eta_squared(t.acc1)
summary(t.acc2 <- aov(ACC2 ~ Grp, data = dat.acc))
1/anovaBF(ACC2 ~ Grp, data = dat.acc)
effectsize::eta_squared(t.acc2)
summary(t.age <- aov(age ~ Grp, data = dat.acc))
1/anovaBF(age ~ Grp, data = dat.acc)
effectsize::eta_squared(t.age)
save(t.acc1, t.acc2, t.age, file = file.path(OutPath, 'ACC_ttest.rd'))

# define new subject numbers
s<-0
dat$Sub <- rep(0,nrow(dat))
for (subj in unique(dat$SubName)) {
  s<- s+1
  dat$Sub[which(dat$SubName==subj)] <- s
}

rgrs <- c('spk','ads')
Contr <- c('PrmAns1','ReqAns2')
#### define group names
GrpNames <- c('alpha', 'sham', 'beta')
GroupNames <- c('10 Hz tACS', 'Sham stimulation', '20 Hz tACS')
####
outgrp <- file.path(OutPath,sprintf('SPS_%imodel_hlogit_group.txt', LevNum))
outsub <- file.path(OutPath,sprintf('SPS_%imodel_hlogit_subject.txt', LevNum))
if (! (file.exists(outgrp) & file.exists(outsub))) {
  Res.g <- data.frame()
  Res.s <- data.frame()
  for (cc in 1:2) {
    if (cc==1) con <- c(3,1)
    if (cc==2) con <- c(4,2)
    ConName <- Contr[cc]
    dat.c <- subset(dat,speech_act_no %in% con)
    dat.sub <- unique(data.frame(Sub=dat.c$Sub,SubName=dat.c$SubName,Grp=dat.c$Grp))
    # make data list for stan
    attach(dat.c)
    X_t <- cbind(rep(1,nrow(dat.c)),spk,ads,guessc)
    dataList <- list(S=nrow(dat.sub),
                    J=length(unique(dat.sub$Grp)),
                    N=nrow(dat.c),
                    K=ncol(X_t),
                    grp=dat.sub$Grp,
                    sub=Sub,
                    y=y,
                    X=X_t
                    )
    detach(dat.c)
    outrda <- file.path(OutPath,sprintf('SPS_%imodel_%s.rd', LevNum, ConName))
    if (!file.exists(outrda)) {
      fit <- stan(modelFile,
                data    = dataList,
                chains  = nChains,
                iter    = nIter,
                cores   = nChains,
                warmup  = nWarmup,
                thin    = nThin,
                init    = 'random',
                control = list(adapt_delta = 0.99, max_treedepth = 10)
                )
      save(fit,file=outrda)
    } else {
      load(outrda)
    }
    
    sm.g <- summary(fit, pars = c('beta_group'), probs = c(0.025, 0.975))$summary
    sm.s <- summary(fit, pars = c('beta_subject'), probs = c(0.025, 0.975))$summary
    
    for (gg in 1:length(GrpNames)) {
      for (rr in 1:length(rgrs)) {
        res.g <- sm.g[rownames(sm.g) == sprintf('beta_group[%i,%i]', gg, rr + 1),]
        res.g <- data.frame(ConNum = Contr[cc], GrpNum = GrpNames[gg], RatNum = rgrs[rr],
                            Mean = res.g['mean'], SE = res.g['se_mean'], SD = res.g['sd'],
                            CrI_lo = res.g['2.5%'], CrI_hi = res.g['97.5%'],
                            n_eff = res.g['n_eff'], Rhat = res.g['Rhat'])
        Res.g <- rbind(Res.g, res.g)
        if (gg == 1) {
          for (ss in unique(dat.sub$Sub)) {
            res.s <- sm.s[rownames(sm.s) == sprintf('beta_subject[%i,%i]', ss, rr+1),]
            SubNum <- as.character(dat.sub$SubName[which(dat.sub$Sub == ss)])
            GrpNumber <- as.integer(dat.sub$Grp[which(dat.sub$Sub == ss)])

            as1 <- subset(dat, SubName == SubNum)

            res.s <- cbind(data.frame(SubNum = SubNum, ConNum = Contr[cc],
                                      GrpNum = GrpNumber, RatNum = rgrs[rr]),
                            data.frame(Mean = res.s['mean'], SE = res.s['se_mean'], SD = res.s['sd'],
                                      CrI_lo = res.s['2.5%'], CrI_hi = res.s['97.5%'],
                                      n_eff = res.s['n_eff'], Rhat = res.s['Rhat']))
            Res.s <- rbind(Res.s, res.s)
  }}}}}
  write.table(Res.g, outgrp, sep = '\t', row.names = F)
  write.table(Res.s, outsub, sep = '\t', row.names = F)
} else {
  Res.g <- read.table(outgrp, sep = '\t', header = T)
  Res.s <- read.table(outsub, sep = '\t', header = T)
}

GroupNames <- c('20 Hz tACS', '10 Hz tACS', 'Sham stimulation')
RatNames <- c('Speaker\'s will', 'Addressee\'s will')
### plot
fb <- Res.g
fb$ConNum <- factor(fb$ConNum,levels=c('PrmAns1','ReqAns2'))
levels(fb$ConNum)[levels(fb$ConNum)=='PrmAns1'] <- 'Promise vs. Reply-1'
levels(fb$ConNum)[levels(fb$ConNum)=='ReqAns2'] <- 'Request vs. Reply-2'
fb$GrpNum <- factor(fb$GrpNum, levels = c(GrpNames[3], GrpNames[1], GrpNames[2]))
levels(fb$GrpNum)[levels(fb$GrpNum)=='beta'] <- GroupNames[1]
levels(fb$GrpNum)[levels(fb$GrpNum)=='alpha'] <- GroupNames[2]
levels(fb$GrpNum)[levels(fb$GrpNum)=='sham'] <- GroupNames[3]
fb$RatNum <- factor(fb$RatNum,levels=c('spk','ads'))
levels(fb$RatNum)[levels(fb$RatNum)=='spk'] <- RatNames[1]
levels(fb$RatNum)[levels(fb$RatNum)=='ads'] <- RatNames[2]
fb$RatNum1 <- ifelse(fb$RatNum==RatNames[1], 1, 2)
str(fb)

outaovt <- file.path(OutPath,sprintf('SPS_%imodel_aovt.txt',LevNum))
if (! file.exists(outaovt)) {
  res.p <- subset(Res.s,GrpNum%in%c(3,1,2))
  res.p$GrpNum <- factor(res.p$GrpNum,levels=c(3,1,2))
  Contr <- c('PrmAns1','ReqAns2')
  rgrs <- c('spk','ads') # RatNames
  Res.test <- data.frame()
  for (cc in 1:2) {
    ConName <- Contr[cc]
    for (rr in 1:length(rgrs)) {
      res.c <- subset(res.p, ConNum==ConName&RatNum==rgrs[rr])
      ## comparisons
      san0 <- summary(a0 <- aov(Mean ~ GrpNum, data = res.c))
      peta2 <- effectsize::eta_squared(a0)
      f0 <- san0[[1]]$`F value`[1]
      df0 <- san0[[1]]$Df[1]
      p0 <- san0[[1]]$`Pr(>F)`[1]
      b0 <- anovaBF(Mean ~ GrpNum, data = res.c, progress=FALSE)
      bf0 <- extractBF(b0)$bf
      t1 <- t.test(x=subset(res.c,GrpNum==3)$Mean,y=subset(res.c,GrpNum==2)$Mean,paired=F)
      d1 <- effectsize::cohens_d(t1)
      b1 <- ttestBF(x=subset(res.c,GrpNum==3)$Mean,y=subset(res.c,GrpNum==2)$Mean,paired=F)
      bf1 <- extractBF(b1)$bf
      t2 <- t.test(x=subset(res.c,GrpNum==1)$Mean,y=subset(res.c,GrpNum==2)$Mean,paired=F)
      d2 <- effectsize::cohens_d(t2)
      b2 <- ttestBF(x=subset(res.c,GrpNum==1)$Mean,y=subset(res.c,GrpNum==2)$Mean,paired=F)
      bf2 <- extractBF(b2)$bf
      t3 <- t.test(x=subset(res.c,GrpNum==3)$Mean,y=subset(res.c,GrpNum==1)$Mean,paired=F)
      d3 <- effectsize::cohens_d(t3)
      b3 <- ttestBF(x=subset(res.c,GrpNum==3)$Mean,y=subset(res.c,GrpNum==1)$Mean,paired=F)
      bf3 <- extractBF(b3)$bf
      res.test <- data.frame(ConNum=ConName, RatNum=rgrs[rr],
                            F_aov=f0, df_aov=df0, eta2 = peta2$Eta2, p_aov=p0, bf_aov=bf0, 
                            t_bs=t1$statistic, df_bs=t1$parameter, d_bs=d1$Cohens_d, p_bs=t1$p.value, bf_bs=bf1,
                            t_as=t2$statistic, df_as=t2$parameter, d_as=d2$Cohens_d, p_as=t2$p.value, bf_as=bf2,
                            t_ba=t3$statistic, df_ba=t3$parameter, d_ba=d3$Cohens_d, p_ba=t3$p.value, bf_ba=bf3
                            )
      Res.test <- rbind(Res.test,res.test)
    }}
  write.table(Res.test,outaovt,sep='\t',row.names=F)
}

###
fb.s <- Res.s
fb.s$GrpNum <- factor(fb.s$GrpNum,levels=c(3,1,2))
levels(fb.s$GrpNum)[levels(fb.s$GrpNum)==3] <- GroupNames[1]
levels(fb.s$GrpNum)[levels(fb.s$GrpNum)==1] <- GroupNames[2]
levels(fb.s$GrpNum)[levels(fb.s$GrpNum)==2] <- GroupNames[3]
fb.s$RatNum <- factor(fb.s$RatNum,levels=c('spk','ads'))
levels(fb.s$RatNum)[levels(fb.s$RatNum)=='spk'] <- RatNames[1]
levels(fb.s$RatNum)[levels(fb.s$RatNum)=='ads'] <- RatNames[2]
fb.s$RatNum1 <- ifelse(fb.s$RatNum==RatNames[1], 1, 2)
fb.s$GrpNum1 <- 0
for (i in 1:3) fb.s$GrpNum1[fb.s$GrpNum==GroupNames[i]] <- i
fb.s$ConNum <- factor(fb.s$ConNum,levels=c('PrmAns1','ReqAns2'))
levels(fb.s$ConNum)[levels(fb.s$ConNum)=='PrmAns1'] <- 'Promise vs. Reply-1'
levels(fb.s$ConNum)[levels(fb.s$ConNum)=='ReqAns2'] <- 'Request vs. Reply-2'

fb.s$Mean.g <- 0
fb.s$CrI_hi.g <- 0
fb.s$CrI_lo.g <- 0
for (cc in unique(fb$ConNum)) {
  for (gg in unique(fb$GrpNum)) {
    for (rr in unique(fb$RatNum)) {
      fb.s$Mean.g[fb.s$ConNum==cc&fb.s$GrpNum==gg&fb.s$RatNum==rr] <- fb$Mean[fb$ConNum==cc&fb$GrpNum==gg&fb$RatNum==rr]
      fb.s$CrI_hi.g[fb.s$ConNum==cc&fb.s$GrpNum==gg&fb.s$RatNum==rr] <- fb$CrI_hi[fb$ConNum==cc&fb$GrpNum==gg&fb$RatNum==rr]
      fb.s$CrI_lo.g[fb.s$ConNum==cc&fb.s$GrpNum==gg&fb.s$RatNum==rr] <- fb$CrI_lo[fb$ConNum==cc&fb$GrpNum==gg&fb$RatNum==rr]
    }}}
str(fb.s)

ggplot(fb.s) + 
  geom_point(aes(x=GrpNum1, y=Mean.g,colour=ConNum),shape=16,size=2.5) +
  theme_bw() + 	theme(panel.grid.major=element_line(colour=NA),
                      panel.grid.minor = element_blank()) +
  theme(panel.grid.major=element_line(colour=NA)) +
  geom_errorbar(aes(x=GrpNum1, y=Mean.g,ymin=CrI_lo.g,ymax=CrI_hi.g,colour=ConNum),width=.2,show.legend = F) +
  geom_point(aes(x=GrpNum1+rnorm(nrow(fb.s),0,0.03), y=Mean,colour=ConNum),
             shape=1, size=rel(1.3), stroke=rel(1.5), alpha=0.6,
             position=position_nudge(x=-0.25)) +
  #############
facet_grid(ConNum~RatNum) +
  scale_x_continuous(breaks=c(1,2,3), labels=GroupNames) +
  scale_colour_discrete(guide='none') +
  guides(linetype=guide_legend(title=NULL)) +
  geom_hline(yintercept=0, colour="black", linetype="longdash") +
  xlab('Group') + ylab('Posterior estimate') +
  theme(axis.text.x=element_text(angle=30,hjust=1,size=rel(1.4),colour='black'),
        axis.title.x=element_text(size=rel(1.3),margin=margin(t=5, r=10, b=0, l=0)),
        axis.text.y=element_text(size=rel(1.4),colour='black'),
        axis.title.y=element_text(size=rel(1.3)),#,margin=margin(t=20, r=0, b=0, l=0)),
        strip.text=element_text(size=rel(1.2)))

ggsave(file.path(OutPath,'SPS_2model_hlogit_group_plot_est95_grp.png'),width=7.5,height=5.9)
