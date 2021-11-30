
library(tidyverse)
library(ggplot2)
library(car)

indiv_data <- read.csv("Fish_Individual_Values.csv")
indiv_data <- na.omit(indiv_data)

indiv_data <- indiv_data %>% mutate(Flow = ifelse(Flow == "0", "Flow 0", "Flow 2")) %>%
                             mutate(Ablation = ifelse(Ablation == "N", "No Ablation", "Ablated")) %>%
                             mutate(Darkness = ifelse(Darkness == "N", "Light", "Dark"))

m_speed <- aov(Speed ~ Flow + Ablation + Darkness + Flow:Ablation + Flow:Darkness, data = indiv_data)
Anova(m_speed)
TukeyHSD(m_speed)

ggplot(indiv_data,aes(x=Speed))+
  geom_histogram() +
  facet_wrap(~ Flow + Ablation + Darkness, scales="free") +
  xlim(0,6) +
  theme_light()

m_heading <- aov(Heading ~ Flow + Ablation + Darkness + Flow*Ablation + Flow*Darkness, data = indiv_data)
Anova(m_heading)
TukeyHSD(m_heading)

ggplot(indiv_data,aes(x=Heading))+
  geom_histogram(binwidth = 10)+
  #coord_polar(theta = "x", start = pi) +
  facet_wrap(~ Flow + Ablation + Darkness, scales = "free") +
  scale_x_continuous(breaks=seq(-180,180,30)) +
  theme_light()

m_tailbeats <- aov(TB_Frequency ~ Flow + Ablation + Darkness + Flow*Ablation + Flow*Darkness, data = indiv_data)
Anova(m_tailbeats)
TukeyHSD(m_tailbeats)

ggplot(indiv_data,aes(x=TB_Frequency))+
  geom_histogram() +
  facet_wrap(~ Flow + Ablation + Darkness, scales="free") +
  xlim(0,6) +
  theme_light()

