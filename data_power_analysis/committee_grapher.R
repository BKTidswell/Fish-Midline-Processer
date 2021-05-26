library(ggplot2)
library(tidyverse)
library(reticulate)
library(CircStats)

# Sys.setenv(PATH= paste("/Users/Ben/opt/anaconda3/envs/R_env/Library/bin",Sys.getenv()["PATH"],sep=";"))
# Sys.setenv(RETICULATE_PYTHON = "/Users/Ben/opt/anaconda3/envs/R_env/python.exe")
# use_condaenv("R_env")
# py_config()
# source_python('circ.py')

data <- read.csv("r_power_data.csv")
close_obs <- data %>% filter(distBin < 3 & angleBin != 6)

rad2deg <- function(rad) {(rad * 180) / (pi)}
deg2rad <- function(deg) {(deg * pi) / (180)}

reshaped_data <- close_obs %>% 
                    select(-c(angleBinSize,distBinSize)) %>%
                    pivot_longer(!c(cond,distBin,angleBin), names_to = "data_type", values_to = "value") %>%
                    mutate(key = paste(cond, distBin, angleBin,sep = "_"))

percent_key <- close_obs %>% group_by(cond) %>%
                             mutate(total = n()) %>%
                             ungroup() %>%
                             group_by(cond, distBin, angleBin) %>% 
                             summarise(percent = n()/mean(total)*100) %>%
                             ungroup() %>%
                             mutate(key = paste(cond, distBin, angleBin,sep = "_")) %>%
                             select(-c(cond, distBin, angleBin))

density_data <- left_join(reshaped_data, percent_key, by = ("key"))



heading_data <- density_data %>% filter(data_type == "heading") #%>%
                                 #mutate(offpara = 90 - abs(90 - value))
                                 #mutate(offpara = ifelse(value <= 90, value, abs(value - 180)))

circ.summary(heading_data$value)$mean

#https://math.stackexchange.com/questions/2154174/calculating-the-standard-deviation-of-a-circular-quantity
heading_sum <- heading_data %>% group_by(cond, distBin, angleBin, percent) %>%
                                #summarise(mean_off = mean(offpara), sd_off = sd(offpara))
                                summarise(mean_angle = rad2deg(atan2(mean( sin( deg2rad(value*4%%360) ) ),
                                                                     mean( cos( deg2rad(value*4%%360) ) ) ))/4,
                                          sd_angle = rad2deg(sqrt(log(1/(mean( sin( deg2rad(value*4%%360) ) )^2 +
                                                                         mean( cos( deg2rad(value*4%%360) ) )^2 ))))/4)
                                
                                
ggplot(heading_data, aes(x = percent, y = value, color = cond)) +
  geom_point(alpha = 0.25) +
  theme_light()

ggplot(heading_sum, aes(x = percent, y = mean_angle, color = cond)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  geom_errorbar(aes(ymin=mean_angle-sd_angle, ymax=mean_angle+sd_angle),width=.2) +
  theme_light()+
  ylab("Heading Difference (Degrees)") +
  xlab("Preference (%)") +
  labs(colour = "Flow Speed\n(BL/s)") +
  scale_color_hue(labels = c("0", "2"))

fit_heading <- glm(value~percent*cond, data = heading_data)
summary(fit_heading)



coord_data <- density_data %>% filter(data_type == "coord")

coord_sum <- coord_data %>% group_by(cond, distBin, angleBin, percent) %>%
                            summarise(mean_coord = mean(value), sd_coord = sd(value))

ggplot(coord_data, aes(x = percent, y = value, color = cond)) +
  geom_point(alpha = 0.25) +
  theme_light()

ggplot(coord_sum, aes(x = percent, y = mean_coord, color = cond)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  theme_light()+
  ylab("Tailbeat Synchonization") +
  xlab("Observations in Area (%)") +
  labs(colour = "Flow Speed\n(BL/s)") +
  scale_color_hue(labels = c("0", "2"))
  

fit <- lm(value~percent, data = coord_data)
Anova(fit)

fit_heading_no_flow <- glm(value~percent, data = heading_data %>% filter(cond == "F0"))
summary(fit_heading_no_flow)

fit_heading_flow <- glm(value~percent, data = heading_data %>% filter(cond == "F2"))
summary(fit_heading_flow)


fit_coord_no_flow <- glm(value~percent, data = coord_data %>% filter(cond == "F0"))
summary(fit_coord_no_flow)

fit_coord_flow <- glm(value~percent, data = coord_data %>% filter(cond == "F2"))
summary(fit_coord_flow)

