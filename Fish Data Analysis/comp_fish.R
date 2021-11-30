library(tidyverse)
library(ggplot2)
library(car)
library(mgcv)
library(viridis)

rad2deg <- function(rad) {(rad * 180) / (pi)}
deg2rad <- function(deg) {(deg * pi) / (180)}
round_any <- function(x, accuracy, f=round){f(x/ accuracy) * accuracy}
ang_mean <- function(x){rad2deg(atan(mean(sin(deg2rad(x)))/mean(cos(deg2rad(x)))))}

comp_data <- read.csv("Fish_Comp_Values.csv")
comp_data <- na.omit(comp_data)

comp_data <- comp_data %>% mutate(Flow = ifelse(Flow == "0", "Flow 0", "Flow 2")) %>%
                           mutate(Abalation = ifelse(Abalation == "N", "No Ablation", "Ablated")) %>%
                           mutate(Darkness = ifelse(Darkness == "N", "Light", "Dark")) %>%
                           mutate(Speed_Diff = abs(Speed_Diff)) %>%
                           mutate(Heading_Diff = abs(Heading_Diff)) %>%
                           filter(abs(X_Distance) <= 3) %>%
                           filter(abs(Y_Distance) <= 3) %>%
                           filter(Speed_Diff <= 5) %>%
                           rename(Ablation = Abalation) %>%
                           rename(Sync = Tailbeat_Offset_Change)

sum_comp_data <- comp_data %>% mutate(X_Distance = round_any(X_Distance,0.25), Y_Distance = round_any(abs(Y_Distance),0.25)) %>%
                               group_by(Flow,Ablation,Darkness,X_Distance,Y_Distance) %>%
                               summarise(Speed_Diff = mean(Speed_Diff),Heading_Diff = ang_mean(Heading_Diff),Sync = mean(Sync))

ggplot(comp_data , aes(x = X_Distance, y = abs(Y_Distance)))+
  stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  scale_fill_viridis() +
  facet_wrap(~ Flow + Ablation + Darkness) +
  xlim(-3,3) + 
  ylim(0,3) +
  theme_light()

#Dataframe for Predictions

x <- seq(from = -3, to = 3, by = 0.25)
y <- seq(from = 0, to = 3, by = 0.25)
flows <- c("Flow 0", "Flow 2")
ablation <- c("No Ablation", "Ablated")
dark <- c("Light","Dark")

predict_df <- expand.grid(X_Distance = x, Y_Distance = y, Flow = flows, Ablation = ablation, Darkness = dark)
predict_df <- predict_df %>% mutate(Distance = sqrt(X_Distance**2 + Y_Distance**2), Angle = rad2deg(atan(Y_Distance/X_Distance)))
predict_df <- predict_df %>% filter(!(Ablation == "Ablated" & Darkness == 'Dark'))
predict_df <- na.omit(predict_df)

## Speed Differences

speed_gam <- gam(Speed_Diff ~ s(Distance)+s(Angle)+s(Distance,Angle)+Flow+Darkness+Ablation, data = comp_data)
summary(speed_gam)
plot(speed_gam,pages=1,residuals=TRUE)
speed_pred <- predict_df %>% mutate(Speed_Diff = predict.gam(speed_gam,predict_df))

ggplot(speed_pred, aes(x = Distance, y = Speed_Diff))+
  geom_smooth()+
  scale_fill_viridis() +
  facet_wrap(~ Flow + Ablation + Darkness) +
  theme_light()

ggplot(speed_pred, aes(x = Angle, y = Speed_Diff))+
  geom_smooth()+
  scale_fill_viridis() +
  facet_wrap(~ Flow + Ablation + Darkness) +
  theme_light()

ggplot(speed_pred, aes(x = X_Distance, y = Y_Distance, fill = Speed_Diff))+
  geom_tile()+
  scale_fill_viridis() +
  facet_wrap(~ Flow + Ablation + Darkness) +
  theme_light()

ggplot(sum_comp_data , aes(x = X_Distance, y = Y_Distance, fill = Speed_Diff))+
  geom_tile() +
  scale_fill_viridis() +
  facet_wrap(~ Flow + Ablation + Darkness) +
  xlim(-3,3) + 
  ylim(0,3) +
  theme_light()

## Heading Differences

heading_gam <- gam(Heading_Diff ~ s(Distance)+s(Angle)+s(Distance,Angle)+Flow+Darkness+Ablation, data = comp_data)
summary(heading_gam)
plot(heading_gam,pages=1,residuals=TRUE)
heading_pred <- predict_df %>% mutate(Heading_Diff = predict.gam(heading_gam,predict_df))

ggplot(heading_pred, aes(x = Distance, y = Heading_Diff))+
  geom_smooth()+
  scale_fill_viridis() +
  facet_wrap(~ Flow + Ablation + Darkness) +
  theme_light()

ggplot(heading_pred, aes(x = Angle, y = Heading_Diff))+
  geom_smooth()+
  scale_fill_viridis() +
  facet_wrap(~ Flow + Ablation + Darkness) +
  theme_light()

ggplot(heading_pred, aes(x = X_Distance, y = Y_Distance, fill = Heading_Diff))+
  geom_tile()+
  scale_fill_viridis() +
  facet_wrap(~ Flow + Ablation + Darkness) +
  theme_light()

ggplot(sum_comp_data , aes(x = X_Distance, y = Y_Distance, fill = abs(Heading_Diff)))+
  geom_tile() +
  scale_fill_viridis() +
  facet_wrap(~ Flow + Ablation + Darkness) +
  xlim(-3,3) + 
  ylim(0,3) +
  theme_light()

## Sync Differences

sync_gam <- gam(Sync ~ s(Distance)+s(Angle)+s(Distance,Angle)+Flow+Darkness+Ablation, data = comp_data)
summary(sync_gam)
plot(sync_gam,pages=1,residuals=TRUE)
sync_pred <- predict_df %>% mutate(Sync = predict.gam(sync_gam,predict_df))

ggplot(sync_pred, aes(x = Distance, y = Sync))+
  geom_smooth()+
  scale_fill_viridis() +
  facet_wrap(~ Flow + Ablation + Darkness) +
  theme_light()

ggplot(sync_pred, aes(x = Angle, y = Sync))+
  geom_smooth()+
  scale_fill_viridis() +
  facet_wrap(~ Flow + Ablation + Darkness) +
  theme_light()

ggplot(sync_pred, aes(x = X_Distance, y = Y_Distance, fill = Sync))+
  geom_tile()+
  scale_fill_viridis() +
  facet_wrap(~ Flow + Ablation + Darkness) +
  theme_light()

ggplot(sum_comp_data , aes(x = X_Distance, y = Y_Distance, fill = abs(Sync)))+
  geom_tile() +
  scale_fill_viridis() +
  facet_wrap(~ Flow + Ablation + Darkness) +
  xlim(-3,3) + 
  ylim(0,3) +
  theme_light()
