library(ggplot2)
library(tidyverse)
library(pwr)
library(lme4)     # for lmer()
library(car)      # for Anova() -- does not actually do ANOVA
library(emmeans)  # for multiple comparisons
library(bbmle)    # for ICtab()
library(betareg)  # for betareg()

power_data <- read.csv("r_power_data.csv")

power_data_close <- power_data %>% filter(distBin < 3 & angleBin != 6)

power_data_close_reshape <- power_data_close %>% 
  select(-c(angleBinSize,distBinSize)) %>%
  pivot_longer(!c(cond,distBin,angleBin), names_to = "data_type", values_to = "value")

# data wrangling 

fish <- power_data_close_reshape %>%
  select(flow = cond, 
         distance = distBin, 
         angle = angleBin, 
         value, 
         data_type) %>%
  mutate(flow = factor(flow, labels = c("no flow", "flow")),
         distance = as.factor(distance),
         angle = as.factor(angle),
         )

# fish headings 

fish_head <- fish %>% 
  subset(data_type == "heading") %>%
  select(-data_type) %>%
  mutate(value = value/180)

ggplot(fish_head, aes(x = value)) + 
  geom_histogram() + 
  facet_wrap(~flow, ncol = 1) +
  theme_bw() + labs(x = "heading value")

# definitely not normally distributed, bounded at 0 and 1
# honestly it looks uniformly distributed which is a feat
# aaand makes it hard to analyze, maybe a beta distribution?

m1.norm <- glm(value ~ flow*distance*angle, family = gaussian, data = fish_head)
m1.beta <- betareg(value ~ flow*distance*angle, data = fish_head)
ICtab(m1.norm, m1.beta) # beta distribution wins this round

Anova(m1.beta)
heading_emeans <- emmeans(m1.beta, specs = ~ flow | distance*angle) %>% 
  contrast(method = "pairwise", adjust = "mvt", type = "response") # look up methods to `adjust` p-values

heading_emeans_df <- data.frame(summary(heading_emeans)) %>% select(-c(contrast,df))

sig_dir <- function(pvals,zvals){
  pval_sig <- as.numeric(pvals <= 0.05)
  zvals_pos <- as.numeric(zvals >= 0)
  zvals_neg <- -1*as.numeric(zvals <= 0)
  
  return((zvals_pos+zvals_neg) * pval_sig)
}

heading_emeans_df <- heading_emeans_df %>% mutate(sig_dir = sig_dir(p.value,z.ratio))

heading_emeans_df <- heading_emeans_df %>% mutate(phi = (as.numeric(angle)-1)*30 + 15,
                                              is_sig = ifelse(sig_dir == 0,"No","Yes"),
                                              out_color = ifelse(sig_dir == 0,"black","white"))

ggplot(heading_emeans_df, aes(x=phi, y=distance, fill=is_sig, color = out_color ))+
  geom_tile(color = heading_emeans_df$out_color)+ 
  coord_polar(theta = "x",start = pi/2)+
  scale_x_continuous(limits = c(-180, 180))+
  theme_light()+ 
  scale_fill_manual(values = c("#ffffff", "#000000"))

# fish coordination

fish_cord <- fish %>% 
  subset(data_type == "coord") %>%
  filter(value != 1) %>%
  select(-data_type)

ggplot(fish_cord, aes(x = value)) + 
  geom_histogram() + 
  facet_wrap(~flow, ncol = 1) +
  theme_bw() + labs(x = "coordination value")

ggplot(fish_cord, aes(x = log(value))) + 
  geom_histogram() + 
  facet_wrap(~flow, ncol = 1) +
  theme_bw() + labs(x = "coordination value")

# not normally distributed, looks more Poisson
# BUT we're not dealing with round numbers... 
# not sure what to do, need to ask Elizabeth

m2.norm <- glm(value ~ flow*distance*angle, family = gaussian, data = fish_cord)

Anova(m2.norm) 
coord_emeans <- emmeans(m2.norm, specs = ~ flow | distance*angle) %>% 
  contrast(method = "pairwise", adjust = "mvt", type = "response") # look up methods to `adjust` p-values

coord_emeans_df <- data.frame(summary(coord_emeans)) %>% select(-c(contrast,df))

coord_emeans_df <- coord_emeans_df %>% mutate(sig_dir = sig_dir(p.value,z.ratio))

coord_emeans_df <- coord_emeans_df %>% mutate(phi = (as.numeric(angle)-1)*30 + 15,
                                              is_sig = ifelse(sig_dir == 0,"No","Yes"),
                                              out_color = ifelse(sig_dir == 0,"black","white"))

ggplot(coord_emeans_df, aes(x=phi, y=distance, fill=is_sig, color = out_color ))+
  geom_tile(color = coord_emeans_df$out_color)+ 
  coord_polar(theta = "x",start = pi/2)+
  scale_x_continuous(limits = c(-180, 180))+
  theme_light()+ 
  scale_fill_manual(values = c("#ffffff", "#000000"))

