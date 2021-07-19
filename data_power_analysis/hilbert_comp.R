library(ggplot2)
library(car)
library(bbmle)
library(multcomp)
library(pwr)
library(tidyverse)

data <- na.omit(read.csv("r_power_data.csv"))

ggplot(data %>% filter(coord < 1), aes(x=coord,y=spd_diff))+
  geom_point()+
  theme_light()

cor(data$coord,data$spd_diff)
cor.test(data$coord,data$spd_diff)

ggplot(data %>% filter(tb_off < 0.5), aes(x=coord,y=tb_off))+
  geom_point()+
  theme_light()

cor(data$coord,data$tb_off)
cor.test(data$coord,data$tb_off)

ggplot(data %>% filter(tb_off < 0.5) %>% filter(coord < 1), aes(x=spd_diff,y=tb_off))+
  geom_point()+
  theme_light()

cor(data$spd_diff,data$tb_off)
cor.test(data$spd_diff,data$tb_off)
