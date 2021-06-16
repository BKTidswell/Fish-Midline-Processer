library(ggplot2)
library(car)
library(bbmle)
library(multcomp)
library(pwr)
library(tidyverse)

# library(reticulate)
# library(CircStats)

# Sys.setenv(PATH= paste("/Users/Ben/opt/anaconda3/envs/R_env/Library/bin",Sys.getenv()["PATH"],sep=";"))
# Sys.setenv(RETICULATE_PYTHON = "/Users/Ben/opt/anaconda3/envs/R_env/python.exe")
# use_condaenv("R_env")
# py_config()
# source_python('circ.py')

data <- read.csv("r_power_data.csv")
close_obs <- data %>% filter(distBin < 3 & angleBin != 6)

rad2deg <- function(rad) {((rad * 180) / (pi))%%(360)}
deg2rad <- function(deg) {((deg * pi) / (180))%%(2*pi)}

reshaped_data <- close_obs %>% 
                    dplyr::select(-c(angleBinSize,distBinSize)) %>%
                    pivot_longer(!c(cond,distBin,angleBin,dist_v,angle_v), names_to = "data_type", values_to = "value") %>%
                    mutate(key = paste(cond, distBin, angleBin,sep = "_"))


percent_key <- close_obs %>% group_by(cond) %>%
                             mutate(total = n()) %>%
                             ungroup() %>%
                             group_by(cond, distBin, angleBin) %>% 
                             summarise(percent = n()/mean(total)*100) %>%
                             ungroup() %>%
                             mutate(key = paste(cond, distBin, angleBin,sep = "_")) %>%
                             select(-c(cond, distBin, angleBin))

density_data <- left_join(reshaped_data, percent_key, by = ("key")) #%>%
                # mutate(distBin = factor(distBin, levels = c(0,1,2), labels = c("0 to 1", "1 to 2", "2 to 3"))) %>%
                # mutate(angleBin = factor(angleBin, levels = c(0,1,2,3,4,5), 
                #                                    labels = c("0 to 30", "30 to 60", "60 to 90", "90 to 120", "120 to 150", "150 to 180")))

#https://github.com/scipy/scipy/blob/master/scipy/stats/morestats.py
circmean <- function(samples){
  rad_samp <- deg2rad(samples)
  
  mean_sin <- mean(sin(rad_samp))
  mean_cos <- mean(cos(rad_samp))
  
  mean_atan <- atan2(mean_sin,mean_cos)
  
  return(rad2deg(mean_atan))
}

circstd <- function(samples){
  rad_samp <- deg2rad(samples)
  
  mean_sin <- mean(sin(rad_samp))
  mean_cos <- mean(cos(rad_samp))
  
  R <- sqrt(mean_cos^2 + mean_sin^2)
  
  std <- sqrt(log(1/R^2))
  
  return(rad2deg(std))
}

heading_data <- density_data %>% filter(data_type == "heading") #%>%
                                 #mutate(offpara = 90 - abs(90 - value))
                                 #mutate(offpara = ifelse(value <= 90, value, value - 180))

#https://math.stackexchange.com/questions/2154174/calculating-the-standard-deviation-of-a-circular-quantity
#circ_multi = 4
# heading_sum <- heading_data %>% group_by(cond, distBin, angleBin, percent) %>%
#                                 #summarise(mean_off = mean(offpara), sd_off = sd(offpara))
#                                 summarise(mean_angle = (rad2deg(atan2(mean( sin( deg2rad(value*circ_multi) ) ),
#                                                                      mean( cos( deg2rad(value*circ_multi) ) ) ))/circ_multi),
#                                           sd_angle = (rad2deg(sqrt(log(1/(mean( sin( deg2rad(value*circ_multi) ) )^2 +
#                                                                          mean( cos( deg2rad(value*circ_multi) ) )^2 ))))/circ_multi))
  
                              
heading_sum <- heading_data %>% group_by(cond, distBin, angleBin, percent) %>%
                                       summarise(mean_angle = circmean(value),
                                                 sd_angle = circstd(value)) %>%
                                        ungroup() %>%
                                        mutate(mean_angle = ifelse(mean_angle > 180, abs(mean_angle - 360), mean_angle))
                                
ggplot(heading_data, aes(x = value, fill = cond)) +
  geom_histogram(alpha = 0.5) +
  coord_polar(theta = "x") +
  scale_y_continuous(breaks=seq(0,360,30)) +
  theme_light()

ggplot(heading_sum, aes(x = percent, y = mean_angle, color = cond)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  #geom_errorbar(aes(ymin=mean_angle-sd_angle, ymax=mean_angle+sd_angle),width=.2) +
  theme_light()+
  ylab("Heading Difference (Degrees)") +
  xlab("Preference (%)") +
  labs(colour = "Flow Speed\n(BL/s)") +
  scale_color_hue(labels = c("0", "2"))

ggplot(heading_sum, aes(x = as.factor(distBin), y = mean_angle, color = cond)) +
  geom_boxplot() +
  geom_smooth(method = "lm", se = FALSE) +
  #geom_errorbar(aes(ymin=mean_angle-sd_angle, ymax=mean_angle+sd_angle),width=.2) +
  theme_light()+
  ylab("Heading Difference (Degrees)") +
  xlab("Distance (BL)") +
  scale_x_discrete(labels = c("0 to 1", "1 to 2", "2 to 3")) +
  scale_y_continuous(breaks=seq(0,360,60)) +
  labs(colour = "Flow Speed\n(BL/s)") +
  scale_color_hue(labels = c("0", "2"))



heading_p_fit <- glm(value~percent*cond, data=heading_data)
Anova(heading_p_fit)
heading_d_fit <- glm(value~dist_v*cond, data=heading_data)
Anova(heading_d_fit)
heading_dBin_fit <- glm(value~distBin*cond, data=heading_data)
Anova(heading_dBin_fit)
ICtab(heading_p_fit,heading_d_fit,heading_dBin_fit)

pwr.f2.test(u=2, v=length(heading_d_fit$model$value), f2 = 1 - heading_d_fit$deviance / heading_d_fit$null.deviance, sig.level = 0.05)


heading_mean_p_fit <- glm(mean_angle~percent*cond, data=heading_sum)
Anova(heading_mean_p_fit)
heading_mean_dBin_fit <- glm(mean_angle~distBin*cond, data=heading_sum)
Anova(heading_mean_dBin_fit)
ICtab(heading_mean_p_fit,heading_mean_dBin_fit)


coord_data <- density_data %>% filter(data_type == "coord") %>% filter(value < 1) 

coord_sum <- coord_data %>% group_by(cond, distBin, angleBin, percent) %>%
                            summarise(mean_coord = mean(value), sd_coord = sd(value))

ggplot(coord_data, aes(x = dist_v, y = value, color = cond)) +
  geom_point(alpha = 0.25) +
  geom_smooth(method = "lm", se = FALSE) +
  theme_light()

ggplot(coord_sum, aes(x = percent, y = mean_coord, color = cond)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  theme_light()+
  ylab("Tailbeat Synchonization") +
  xlab("Preference (%)") +
  labs(colour = "Flow Speed\n(BL/s)") +
  scale_color_hue(labels = c("0", "2"))

ggplot(coord_sum, aes(x = as.factor(distBin), y = mean_coord, color = cond)) +
  geom_boxplot() +
  geom_smooth(method = "lm", se = FALSE) +
  #geom_errorbar(aes(ymin=mean_coord-sd_coord, ymax=mean_coord+sd_coord),width=.2) +
  theme_light()+
  ylab("Tailbeat Synchonization") +
  xlab("Distance (BL)") +
  scale_x_discrete(labels = c("0 to 1", "1 to 2", "2 to 3")) +
  labs(colour = "Flow Speed\n(BL/s)") +
  scale_color_hue(labels = c("0", "2"))
  
coord_p_fit <- glm(value~percent*cond, data=coord_data)
Anova(coord_p_fit)
coord_d_fit <- glm(value~dist_v*cond, data=coord_data)
Anova(coord_d_fit)
coord_dBin_fit <- glm(value~distBin*cond, data=coord_data)
Anova(coord_dBin_fit)
ICtab(coord_p_fit,coord_d_fit,coord_dBin_fit)

pwr.f2.test(u=2, v=length(coord_d_fit$model$value), f2 = 1 - coord_d_fit$deviance / coord_d_fit$null.deviance, sig.level = 0.05)


coord_mean_p_fit <- glm(mean_coord~percent*cond, data=coord_sum)
Anova(coord_mean_p_fit)
coord_mean_dBin_fit <- glm(mean_coord~distBin*cond, data=coord_sum)
Anova(coord_mean_dBin_fit)
ICtab(coord_mean_p_fit,coord_mean_dBin_fit)


tbf_data <- density_data %>% filter(data_type == "tbf")

tbf_sum <- tbf_data %>% group_by(cond, distBin, angleBin, percent) %>%
  summarise(mean_tbf = mean(value), sd_tbf = sd(value))

ggplot(tbf_data, aes(x = percent, y = value, color = cond)) +
  geom_point(alpha = 0.25) +
  theme_light()

ggplot(tbf_sum, aes(x = percent, y = mean_tbf, color = cond)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  #geom_errorbar(aes(ymin=mean_tbf-sd_tbf, ymax=mean_tbf+sd_tbf),width=.2) +
  theme_light()+
  ylab("Tailbeat Frequency (Beasts/sec)") +
  xlab("Preference (%)") +
  labs(colour = "Flow Speed\n(BL/s)") +
  scale_color_hue(labels = c("0", "2"))

ggplot(tbf_sum, aes(x = as.factor(distBin), y = mean_tbf, color = cond)) +
  geom_boxplot() +
  geom_smooth(method = "lm", se = FALSE) +
  #geom_errorbar(aes(ymin=mean_angle-sd_angle, ymax=mean_angle+sd_angle),width=.2) +
  theme_light()+
  ylab("Tailbeat Frequency (Beasts/sec)") +
  xlab("Distance (BL)") +
  scale_x_discrete(labels = c("0 to 1", "1 to 2", "2 to 3")) +
  labs(colour = "Flow Speed\n(BL/s)") +
  scale_color_hue(labels = c("0", "2"))


tbf_p_fit <- glm(value~percent*cond, data=tbf_data)
summary(tbf_p_fit)
tbf_d_fit <- glm(value~dist_v*cond, data=tbf_data)
Anova(tbf_d_fit)
tbf_dBin_fit <- glm(value~distBin*cond, data=tbf_data)
Anova(tbf_dBin_fit)
ICtab(tbf_p_fit,tbf_d_fit,tbf_dBin_fit)

pwr.f2.test(u=2, v=length(tbf_d_fit$model$value), f2 = 1 - tbf_d_fit$deviance / tbf_d_fit$null.deviance, sig.level = 0.05)

tbf_mean_p_fit <- glm(mean_tbf~percent*cond, data=tbf_sum)
Anova(tbf_mean_p_fit)
tbf_mean_dBin_fit <- glm(mean_tbf~distBin*cond, data=tbf_sum)
Anova(tbf_mean_dBin_fit)
ICtab(tbf_mean_p_fit,tbf_mean_dBin_fit)


spd_data <- density_data %>% filter(data_type == "spd_diff")

spd_sum <- spd_data %>% group_by(cond, distBin, angleBin, percent) %>%
                          summarise(mean_spd = abs(mean(value)), sd_spd = sd(value))

ggplot(close_obs,aes(x=spd_diff,y=coord))+
  geom_point()

ggplot(spd_data, aes(x = percent, y = abs(value), color = cond)) +
  geom_point(alpha = 0.25) +
  theme_light()

ggplot(spd_sum, aes(x = percent, y = mean_spd, color = cond)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  #geom_errorbar(aes(ymin=mean_tbf-sd_tbf, ymax=mean_tbf+sd_tbf),width=.2) +
  theme_light()+
  ylab("Speed Difference (BL/s)") +
  xlab("Preference (%)") +
  labs(colour = "Flow Speed\n(BL/s)") +
  scale_color_hue(labels = c("0", "2"))

ggplot(spd_sum, aes(x = as.factor(distBin), y = mean_spd, color = cond)) +
  geom_boxplot() +
  geom_smooth(method = "lm", se = FALSE) +
  #geom_errorbar(aes(ymin=mean_angle-sd_angle, ymax=mean_angle+sd_angle),width=.2) +
  theme_light()+
  ylab("Speed Difference (BL/s)") +
  xlab("Distance (BL)") +
  scale_x_discrete(labels = c("0 to 1", "1 to 2", "2 to 3")) +
  labs(colour = "Flow Speed\n(BL/s)") +
  scale_color_hue(labels = c("0", "2"))


spd_p_fit <- glm(value~percent*cond, data=spd_data)
Anova(spd_p_fit)
spd_d_fit <- glm(value~dist_v*cond, data=spd_data)
Anova(spd_d_fit)
spd_dBin_fit <- glm(value~distBin*cond, data=spd_data)
Anova(spd_dBin_fit)
ICtab(spd_p_fit,spd_d_fit,spd_dBin_fit)

spd_mean_p_fit <- glm(mean_spd~percent*cond, data=spd_sum)
Anova(spd_mean_p_fit)
spd_mean_dBin_fit <- glm(mean_spd~distBin*cond, data=spd_sum)
Anova(spd_mean_dBin_fit)
ICtab(spd_mean_p_fit,spd_mean_dBin_fit)
