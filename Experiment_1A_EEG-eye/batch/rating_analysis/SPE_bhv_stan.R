rm(list=ls())
library(rstan)
rstan_options(auto_write = TRUE)
# basic MCMC parameter
nIter     <- 10000
nChains   <- 4
nWarmup   <- floor(nIter/2)
nThin     <- 1
options(mc.cores = nChains)
RootPath <- '~/Experiments/SPE'
BhvPath <- file.path(RootPath,'raw','bhv')
OutPath <- file.path(RootPath, 'output', 'bhv_2model')
if (!dir.exists(OutPath)) dir.create(OutPath)

infile <- file.path(OutPath,'SPE_rating_data.txt')
if (file.exists(infile)) {
    dat <- read.table(infile, sep='\t', header=T)
} else {
    SubjFiles <- dir(BhvPath,'txt')
    SubEx <- c(106, 142)
    dat <- data.frame()
    sct <- 0
    for (ss in 1:length(SubjFiles)) {
        subj <- SubjFiles[ss]
        dat1 <- read.table(file.path(BhvPath,subj),sep='\t',header=T)
        if (dat1$Sub[1] %in% SubEx) next
        dat1 <- subset(dat1,run!=0)
        # compute task accuracy
        dat12 <- dat1
        dat12$AgtAlt_des[which(dat12$AgtAlt_des==-99)] <- 0
        dat12$IsCorrect[which(dat12$IsCorrect==-99)] <- 0
        acc1 <- mean(dat12$AgtAlt_des)
        acc2 <- mean(subset(dat12,probe_if==1)$IsCorrect)
        # subject number
        sct <- sct + 1
        dat1$Subn <- sct
        dat1$ACC1 <- acc1
        dat1$ACC2 <- acc2
        if (sct==2) colnames(dat) <- colnames(dat1)
        dat <- rbind(dat,dat1)
    }
    write.table(dat, file=infile, sep='\t', row.names=F)
}

dat.acc <- aggregate(ACC1 ~ Sub*ACC2, FUN=mean, data=dat)
sprintf('mean = %.2f, sd = %.2f, range: [%.2f, %.2f]', 
        mean(dat.acc$ACC1),
        sd(dat.acc$ACC1),
        min(dat.acc$ACC1),
        max(dat.acc$ACC1))
sprintf('mean = %.2f, sd = %.2f, range: [%.2f, %.2f]', 
        mean(dat.acc$ACC2),
        sd(dat.acc$ACC2),
        min(dat.acc$ACC2),
        max(dat.acc$ACC2))
acc1_lob <- mean(dat.acc$ACC1)-2*sd(dat.acc$ACC1)
acc2_lob <- mean(dat.acc$ACC2)-2*sd(dat.acc$ACC2)
dat.acc$ACC1
dat.acc$ACC2
sprintf('ACC1: mean = %.2f, sd = %.4f; ACC2: mean = %.2f, sd = %.4f',
            mean(dat.acc$ACC1), sd(dat.acc$ACC1), 
            mean(dat.acc$ACC2), sd(dat.acc$ACC2))

file.g <- file.path(OutPath,'SPE_rating_stanmodel_group.txt')
file.s <- file.path(OutPath,'SPE_rating_stanmodel_subject.txt')
if (file.exists(file.g) & file.exists(file.s)) {
    Res.g <- read.table(file.g, header=T)
    Res.s <- read.table(file.s, header=T)
} else {
    Res.s <- data.frame()
    Res.g <- data.frame()
    rgrs <- c('spk','ads')
    Contr <- c('PrmAns1','ReqAns2')
    for (cc in 1:length(Contr)) {
        con <- Contr[cc]
        outrd <- file.path(OutPath, sprintf('Rating_stanmodel_%s.rd',con))
        if (con=='PrmAns1') cnds <- c('promise', 'answer1')
        if (con=='ReqAns2') cnds <- c('request', 'answer2')
        dat.c <- subset(dat, speech_act_type%in%cnds)
        dat.c$y <- NA
        dat.c$y[dat.c$speech_act_type==cnds[1]] <- as.integer(1)
        dat.c$y[dat.c$speech_act_type==cnds[2]] <- as.integer(0)
        SubNames <- data.frame(unique(cbind(dat.c$Sub, dat.c$Subn)))
        colnames(SubNames) <- c('Sub','Subn')
        str(dat.c)
        # make data list for stan
        if (file.exists(outrd)) {
            load(outrd)
        } else {
            attach(dat.c)
            X_t <- cbind(rep(1,nrow(dat.c)),SpkRat_scr,AdsRat_scr)
            dataList <- list(S=nrow(SubNames),
                            J=1,
                            N=nrow(dat.c),
                            K=ncol(X_t),
                            # grp=rep(1,length(SubNames)),
                            # age=Age,
                            sub=dat.c$Subn,
                            y=y,
                            X=X_t
            )
            detach(dat.c)
            
            modelFile <- file.path(RootPath, 'batch/Rstat/model_two_1g.stan')
            fit <- stan(modelFile,
                        data    = dataList,
                        chains  = nChains,
                        iter    = nIter,
                        cores   = nChains,
                        warmup  = nWarmup,
                        thin    = nThin,
                        init    = "random",
                        control = list(adapt_delta = 0.98, max_treedepth = 12),
                        seed    = as.numeric(format(Sys.time(),'%H%M%S%m%d%Y'))[[1]]
            )
            save(fit, file=outrd)
        }
            
        sm.g <- summary(fit, pars=c('beta_group'), probs=c(0.025,0.975))$summary
        sm.s <- summary(fit, pars=c('beta_subject'), probs=c(0.025,0.975))$summary
        # group-level estimates
        for (rr in 1:length(rgrs)) {
            res.g <- sm.g[rownames(sm.g)==sprintf('beta_group[%i]',rr+1),]
            res.g <- data.frame(ConNum=Contr[cc],RatNum=rgrs[rr],
                                Mean=res.g['mean'], SE=res.g['se_mean'], SD=res.g['sd'],
                                CrI_lo=res.g['2.5%'], CrI_hi=res.g['97.5%'],
                                n_eff=res.g['n_eff'], Rhat=res.g['Rhat'])
            Res.g <- rbind(Res.g, res.g)
            # subject-level estimates
            for (ss in unique(dat$Subn)) {
                res.s <- sm.s[rownames(sm.s)==sprintf('beta_subject[%i,%i]',ss,rr+1),]
                SubNum <- as.character(SubNames$Sub[SubNames$Subn==ss])
                
                res.s <- cbind(data.frame(SubNum=SubNum,SS=ss,ConNum=Contr[cc],
                                            RatNum=rgrs[rr]),
                                data.frame(Mean=res.s['mean'], SE=res.s['se_mean'], SD=res.s['sd'],
                                            CrI_lo=res.s['2.5%'], CrI_hi=res.s['97.5%'],
                                            n_eff=res.s['n_eff'], Rhat=res.s['Rhat']))
                Res.s <- rbind(Res.s, res.s)
    }}}
    write.table(Res.g, file.g, sep='\t', row.names=F)
    write.table(Res.s, file.s, sep='\t', row.names=F)
}
str(Res.g)
str(Res.s)
### plot
Res.g$ConNum <- factor(Res.g$ConNum, levels=c('PrmAns1','ReqAns2'))
levels(Res.g$ConNum)[levels(Res.g$ConNum)=='PrmAns1'] <- 'Promise vs. Reply-1'
levels(Res.g$ConNum)[levels(Res.g$ConNum)=='ReqAns2'] <- 'Request vs. Reply-2'
Res.g$RatNum1 <- 0
Res.g$RatNum1[Res.g$RatNum=='spk'] <- 1
Res.g$RatNum1[Res.g$RatNum=='ads'] <- 2
str(Res.g)

Res.s$RatNum1 <- 0
Res.s$RatNum1[Res.s$RatNum=='spk'] <- 1
Res.s$RatNum1[Res.s$RatNum=='ads'] <- 2
Res.s$ConNum <- factor(Res.s$ConNum, levels=c('PrmAns1','ReqAns2'))
levels(Res.s$ConNum)[levels(Res.s$ConNum)=='PrmAns1'] <- 'Promise vs. Reply-1'
levels(Res.s$ConNum)[levels(Res.s$ConNum)=='ReqAns2'] <- 'Request vs. Reply-2'
str(Res.s)

Res.s$Mean.g <- 0
Res.s$CrI_hi.g <- 0
Res.s$CrI_lo.g <- 0
for (cc in unique(Res.g$ConNum)) {
    for (rr in unique(Res.g$RatNum)) {
      Res.s$Mean.g[Res.s$ConNum==cc&Res.s$RatNum==rr] <- Res.g$Mean[Res.g$ConNum==cc&Res.g$RatNum==rr]
      Res.s$CrI_hi.g[Res.s$ConNum==cc&Res.s$RatNum==rr] <- Res.g$CrI_hi[Res.g$ConNum==cc&Res.g$RatNum==rr]
      Res.s$CrI_lo.g[Res.s$ConNum==cc&Res.s$RatNum==rr] <- Res.g$CrI_lo[Res.g$ConNum==cc&Res.g$RatNum==rr]
}}

pp <- ggplot(Res.s) + 
  geom_point(aes(x=RatNum1, y=Mean.g,colour=ConNum),shape=16,size=2.5) +
  theme_bw() + 	theme(panel.grid.major=element_line(colour=NA),
                      panel.background = element_rect(fill = "transparent",colour = NA),
                      plot.background = element_rect(fill = "transparent",colour = NA),
                      panel.grid.minor = element_blank()) +
  theme(panel.grid.major=element_line(colour=NA)) +
  geom_errorbar(aes(x=RatNum1, y=Mean.g,ymin=CrI_lo.g,ymax=CrI_hi.g,colour=ConNum),width=.2,show.legend = F) +
  geom_point(aes(x=RatNum1+rnorm(nrow(Res.s),0,0.03), y=Mean,colour=ConNum),
              shape=1, size=rel(1.3), stroke=rel(1.5), alpha=0.6,
              position=position_nudge(x=-0.25)) +
  #############
  facet_grid(.~ConNum) + 
  scale_x_continuous(breaks=c(1,2),
                   labels=c('Speaker\'s will','Addressee\'s will')) +
  scale_colour_discrete(guide='none') +
  guides(linetype=guide_legend(title=NULL)) +
  geom_hline(yintercept=0, colour="black", linetype="longdash") +
  xlab('Ratings') + ylab('Posterior estimate') +
  theme(axis.text.x=element_text(angle=40,hjust=0.99,size=rel(1.4),colour='black'),
        axis.title.x=element_text(size=rel(1.3),margin=margin(t=5, r=10, b=0, l=0)),
        axis.text.y=element_text(size=rel(1.4),colour='black'),
        axis.title.y=element_text(size=rel(1.3)),#,margin=margin(t=20, r=0, b=0, l=0)),
        strip.text=element_text(size=rel(1.2)))

ggsave(file.path(OutPath,'SPE_rating_stanmodel.png'),plot=pp,width=7,height=5.5)
ggsave(file.path(OutPath,'SPE_rating_stanmodel.pdf'),plot=pp,width=7,height=5.5)
