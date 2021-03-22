library(ggplot2)
library(tidyverse)
library(pwr)

power_data <- read.csv("r_power_data.csv")

power_data_close <- power_data %>% filter(distBin < 3 & angleBin != 6)

power_data_close_flow <- power_data_close %>% filter(cond == "F2")
power_data_close_no_flow <- power_data_close %>% filter(cond == "F0")

var(power_data_close$heading)
var(power_data_close$coord)

power_data_summary <- power_data_close %>% group_by(cond,distBin,angleBin) %>%
                                           summarise(headingSD = sd(heading), headingMean = mean(heading),
                                                     coordSD = sd(coord), coordMean = mean(coord),
                                                     headingVar = var(heading), coordVar = var(coord))

#Heading First

eHead <- abs(mean(power_data_close_flow$heading) - mean(power_data_close_no_flow$heading)) / var(power_data_close$heading)
n1Head <- length(power_data_close_flow$cond)
n2Head <- length(power_data_close_no_flow$cond)

pwr.t2n.test(n1 = as.integer(n1Head), n2 = as.integer(n2Head), d = eHead, sig.level = 0.05)


#Coord next

eCord <- abs(mean(power_data_close_flow$coord) - mean(power_data_close_no_flow$coord)) / var(power_data_close$coord)
n1Cord <- length(power_data_close_flow$cond)
n2Cord <- length(power_data_close_no_flow$cond)

pwr.t2n.test(n1 = as.integer(n1Cord), n2 = as.integer(n2Cord), d = eCord, sig.level = 0.05)


#Anova??

power_data_close$angleBin <- as.factor(power_data_close$angleBin)
power_data_close$distBin <- as.factor(power_data_close$distBin)

headingAOV <- aov(heading ~ cond:angleBin:distBin, data = power_data_close)
summary(headingAOV)
TukeyHSD(headingAOV)

##Changing them for lots of separate power tests per square

power_data_close_flow <- power_data_close_flow %>% 
                          select(-c(angleBinSize,distBinSize)) %>%
                          pivot_longer(!c(cond,distBin,angleBin), names_to = "data_type", values_to = "value")


power_data_close_flow_sum <- power_data_close_flow %>% 
                              group_by(cond,distBin,angleBin,data_type) %>%
                              summarise(sd = sd(value), mean = mean(value), n = length(value))

power_data_close_no_flow <- power_data_close_no_flow %>% 
                              select(-c(angleBinSize,distBinSize)) %>%
                              pivot_longer(!c(cond,distBin,angleBin), names_to = "data_type", values_to = "value")

power_data_close_no_flow_sum <- power_data_close_no_flow %>% 
                                  group_by(cond,distBin,angleBin,data_type) %>%
                                  summarise(sd = sd(value), mean = mean(value), n = length(value))

#Now join them
joined_df <- left_join(power_data_close_flow_sum,
                       power_data_close_no_flow_sum, 
                       by = c("distBin","angleBin","data_type"),
                       suffix = c("_F2", "_F0"))

#Now get one that matches for the var total

power_data_close_reshape <- power_data_close %>% 
                              select(-c(angleBinSize,distBinSize)) %>%
                              pivot_longer(!c(cond,distBin,angleBin), names_to = "data_type", values_to = "value")

power_data_close_reshape_sum <- power_data_close_reshape %>% 
                                  group_by(distBin,angleBin,data_type) %>%
                                  summarise(var = var(value))

power_data_close_reshape_sum$distBin <- as.numeric(power_data_close_reshape_sum$distBin)-1
power_data_close_reshape_sum$angleBin <- as.numeric(power_data_close_reshape_sum$angleBin)-1

#Now join that to get overall var for each box
joined_df <- left_join(joined_df,
                       power_data_close_reshape_sum, 
                       by = c("distBin","angleBin","data_type"))

#Okay so now calculate the power for each bin

joined_df_final <- joined_df %>%
                    mutate(n_F2 = as.integer(n_F2), n_F0 = as.integer(n_F0)) %>%
                    mutate(e = abs(mean_F2 - mean_F0)/var) %>%
                    mutate(power = 100*pwr.t2n.test(n1 = n_F2, n2 = n_F0, d = e, sig.level = 0.05)$power)

ggplot(joined_df_final %>% filter(data_type=="heading"), aes(angleBin, distBin, fill= power)) + 
  geom_tile()+
  ggtitle("Power of Heading Analysis")+
  geom_text(aes(label = round(power,0))) +
  scale_fill_gradient(low = "white", high = "red")+
  theme_light()

ggplot(joined_df_final %>% filter(data_type=="coord"), aes(angleBin, distBin, fill= power)) + 
  geom_tile()+
  ggtitle("Power of Tailbeat Analysis")+
  geom_text(aes(label = round(power,0))) +
  scale_fill_gradient(low = "white", high = "red")+
  theme_light()

#Okay so basically in every place there is more than enough power

#Actually there is not when I normalize for framerate :(

model <- aov(coord ~ distBin*angleBin*cond, data = power_data)
summary(model)


