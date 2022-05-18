import random
import math
import pandas as pd
import plotly.express as px
import numpy as np


#random.seed(-77)
random.seed(77)

tailbeat_len = 19

FISH_BL = 7 #cm

MEAN_SPEED = 1.5 / 60 * FISH_BL #BL/s / 60 to get BL/frame * FISH_BL for cm/frame
SD_SPEED = 0.8 / 60 * FISH_BL

MEAN_TURN = math.pi/4
SD_TURN = math.pi/6

MEAN_COAST = 60
SD_COAST = 15

MIN_X = 0
MIN_Y = 0

MAX_X = 60 #cm
MAX_Y = 25 #cm

def get_dist_np(x1s,y1s,x2s,y2s):
    dist = np.sqrt((x1s-x2s)**2+(y1s-y2s)**2)
    return dist

def directionOfPoint(Cx, Cy, Prex, Prey, Pxs, Pys):

    total_lefts = 0
    total_rights = 0

    for i in range(len(Pxs)):

        #print(Cx, Cy, Prex, Prey, Pxs[i], Pys[i])

        # Subtracting co-ordinates of 
        # point A from B and P, to 
        # make A as origin
        Prex_adj = Prex - Cx
        Prey_adj = Prey - Cy
        Pxs[i] -= Cx
        Pys[i] -= Cx
       
        # Determining cross Product
        cross_product = Prex_adj * Pys[i] - Prey_adj * Pxs[i]
       
        # Return RIGHT if cross product is positive
        if (cross_product > 0):
            total_rights += 1
              
        # Return LEFT if cross product is negative
        if (cross_product < 0):
            total_lefts += 1

    return(total_lefts,total_rights)

class fish_agent:
    def __init__(self,start_x,start_y,start_angle,speed,name):
        self.start_x = start_x
        self.start_y = start_y
        self.start_angle = start_angle
        self.speed = speed
        self.xs = [self.start_x]
        self.ys = [self.start_y]
        self.angles = [self.start_angle]
        self.log = ["Started"]
        self.time = [0]

        self.name = name

        self.move()

    def move(self):
        while(len(self.xs) < 300):

            coast_len = int(random.gauss(MEAN_COAST, SD_COAST))

            for x in range(coast_len):

                new_angle = self.angles[-1]

                next_x = self.xs[-1] + self.speed * math.cos(new_angle)
                next_y = self.ys[-1] + self.speed * math.sin(new_angle)

                message = "Coasting"

                while(next_x < MIN_X or next_x > MAX_X or next_y < MIN_Y or next_y > MAX_Y):
                    angle_change = random.gauss(MEAN_TURN, SD_TURN)

                    if random.random() > 0.5:
                        new_angle = (new_angle + angle_change) % (2*math.pi)
                        message = "left"
                    else:
                        new_angle = (new_angle - angle_change) % (2*math.pi)
                        message = "right"

                    next_x = self.xs[-1] + self.speed * math.cos(new_angle)
                    next_y = self.ys[-1] + self.speed * math.sin(new_angle)

                self.xs.append(next_x)
                self.ys.append(next_y)
                self.angles.append(new_angle)
                self.log.append(message)
                self.time.append(self.time[-1]+1)

            angle_change = random.gauss(MEAN_TURN, SD_TURN)

            if random.random() > 0.5:
                new_angle = (self.angles[-1] + angle_change) % (2*math.pi)
                message = "left"
            else:
                new_angle = (self.angles[-1] - angle_change) % (2*math.pi)
                message = "right"

            self.xs.append(self.xs[-1])
            self.ys.append(self.ys[-1])
            self.angles.append(new_angle)
            self.log.append(message)
            self.time.append(self.time[-1]+1)

        self.xs = np.asarray(self.xs[0:299])
        self.ys = np.asarray(self.ys[0:299])
        self.angles = np.asarray(self.angles[0:299])
        self.log = np.asarray(self.log[0:299])
        self.time = self.time[0:299]

    def output(self):
        d = {'Xs': self.xs, 'Ys': self.ys, "Angles":self.angles, "Log":self.log, "Time": self.time, "Num": self.num}
        df = pd.DataFrame(data=d)
        return(df)

class fish_comp:
    def __init__(self, fish1, fish2):
        self.name = fish1.name + "x" + fish2.name
        self.f1 = fish1
        self.f2 = fish2
        self.x_diff = []
        self.y_diff = []
        self.dist = []
        self.heading_diff = []

        self.calc_dist()
        self.calc_heading_diff()

    def calc_dist(self):        
        #Divided to get it into bodylengths
        self.x_diff = (self.f1.xs - self.f2.xs)/FISH_BL
         #the y_diff is negated so it faces correctly upstream
        self.y_diff = -1*(self.f1.ys - self.f2.ys)/FISH_BL

        self.dist = get_dist_np(0,0,self.x_diff,self.y_diff)

    def calc_heading_diff(self):
        self.heading_diff = np.rad2deg(np.arctan2(np.sin(self.f1.angles-self.f2.angles),
                                                  np.cos(self.f1.angles-self.f2.angles)))

class fish_turns:
    def __init__(self,main_fish,other_fishes):
        self.main_fish = main_fish
        self.other_fish = other_fishes[:]

        self.other_fish.remove(main_fish)

        self.lefts = []
        self.rights = []
        self.frames = []
        self.dirs = []

        self.get_turn_nums()

    def get_turn_nums(self):
        turn_indexes = (np.logical_or(self.main_fish.log == "left",self.main_fish.log == "right")).nonzero()

        for i in turn_indexes[0]:
            other_xs = [of.xs[i] for of in self.other_fish]
            other_ys = [of.ys[i] for of in self.other_fish]

            curr_x = self.main_fish.xs[i]
            curr_y = self.main_fish.ys[i]

            prev_x = self.main_fish.xs[i-5]
            prev_y = self.main_fish.ys[i-5]

            t_lefts,t_rights = directionOfPoint(curr_x,curr_y,prev_x,prev_y,other_xs,other_ys)

            self.lefts.append(t_lefts)
            self.rights.append(t_rights)
            self.frames.append(i)
            self.dirs.append(self.main_fish.log[i])

class trial_comp:
    def __init__(self,n_fish,trial_num,flow):
        self.year = 1995
        self.month = 9
        self.day = 7
        self.trial = trial_num

        self.ablation = "Simulated"
        self.darkness = "Simulated"
        self.flow = flow

        self.n_fish = n_fish

        self.fishes = [fish_agent(random.randint(MIN_X,MAX_X),
                          random.randint(MIN_Y,MAX_Y),
                          random.uniform(0,2*math.pi),
                          random.gauss(MEAN_SPEED,SD_SPEED),
                          "individual{x}".format(x=i)) for i in range(n_fish)]

        self.fish_comp_indexes = [[i,j] for i in range(n_fish) for j in range(i+1,n_fish)]

        for pair in self.fish_comp_indexes:
            random.shuffle(pair)

        self.fish_comps = [[0 for j in range(self.n_fish)] for i in range(self.n_fish)]

        #Now we fill in based on the randomized pairs
        for pair in self.fish_comp_indexes:
            self.fish_comps[pair[0]][pair[1]] = fish_comp(self.fishes[pair[0]],self.fishes[pair[1]])

        self.fish_turns = [fish_turns(f,self.fishes) for f in self.fishes]

    def return_raw_comp_vals(self):
        firstfish = True

        for pair in self.fish_comp_indexes:

            current_comp = self.fish_comps[pair[0]][pair[1]]

            dists = get_dist_np(0,0,current_comp.x_diff,current_comp.y_diff)

            short_data_length = min([len(current_comp.x_diff),len(current_comp.y_diff),len(dists)])

            if short_data_length > tailbeat_len:

                d = {'Year': np.repeat(self.year,short_data_length),
                     'Month': np.repeat(self.month,short_data_length),
                     'Day': np.repeat(self.day,short_data_length),
                     'Trial': np.repeat(self.trial,short_data_length), 
                     'Ablation': np.repeat(self.ablation,short_data_length), 
                     'Darkness': np.repeat(self.darkness,short_data_length), 
                     'Flow': np.repeat(self.flow,short_data_length), 
                     'Fish': np.repeat(current_comp.name,short_data_length),
                     'Frame_Num': range(short_data_length),
                     'X_Distance': current_comp.x_diff[:short_data_length], 
                     'Y_Distance': current_comp.y_diff[:short_data_length], 
                     'Distance': dists[:short_data_length],
                     'Angle': np.repeat(361,short_data_length),
                     'Fish1_Heading': np.rad2deg(current_comp.f1.angles[:short_data_length]),
                     'Fish2_Heading': np.rad2deg(current_comp.f2.angles[:short_data_length]),
                     'Heading_Diff': current_comp.heading_diff[:short_data_length],}

                if firstfish:
                    out_data = pd.DataFrame(data=d)
                    firstfish = False
                else:
                    out_data = out_data.append(pd.DataFrame(data=d))

        return(out_data)

    def return_fish_turns(self):
        firstfish = True

        for turn_data in self.fish_turns:

            data_len = len(turn_data.lefts)

            d = {'Year': np.repeat(self.year,data_len),
                 'Month': np.repeat(self.month,data_len),
                 'Day': np.repeat(self.day,data_len),
                 'Trial': np.repeat(self.trial,data_len), 
                 'Ablation': np.repeat(self.ablation,data_len), 
                 'Darkness': np.repeat(self.darkness,data_len), 
                 'Flow': np.repeat(self.flow,data_len), 
                 'Turn Start Frame': np.repeat(9,data_len),
                 'Turn End Frame': turn_data.frames,
                 'Fish': np.repeat(turn_data.main_fish.name,data_len),
                 'Turn Direction': turn_data.dirs,
                 'Appox Turn Angle': np.repeat(361,data_len),
                 'No. Fish Left': turn_data.lefts,
                 'No. Fish Right': turn_data.rights}

            if firstfish:
                out_data = pd.DataFrame(data=d)
                firstfish = False
            else:
                out_data = out_data.append(pd.DataFrame(data=d))

        return(out_data)


trials = []

for x in range(10):
    trials.append(trial_comp(8,x,"0"))

first_trial = True

#pair = trials[0].fish_comp_indexes[3]
#trials[0].fish_comps[pair[0]][pair[1]].graph_values()

print("Creating CSVs...")

for trial in trials:
    if first_trial:
        fish_raw_comp_dataframe = trial.return_raw_comp_vals()
        fish_turn_dataframe = trial.return_fish_turns()
        first_trial = False
    else:
        fish_raw_comp_dataframe = fish_raw_comp_dataframe.append(trial.return_raw_comp_vals())
        fish_turn_dataframe = fish_turn_dataframe.append(trial.return_fish_turns())

fish_raw_comp_dataframe.to_csv("Fish_Raw_Comp_Values_Sim.csv")
fish_turn_dataframe.to_csv("Fish_Turn_Values_Sim.csv")

# first_fish = True

# for x in range(8):
#     if first_fish:
#         fish = fish_agent(random.randint(MIN_X,MAX_X),
#                           random.randint(MIN_Y,MAX_Y),
#                           random.uniform(0,2*math.pi),
#                           random.gauss(MEAN_SPEED,SD_SPEED),
#                           "fish{x}".format(x=x))
#         fish.move()
#         full_df = fish.output()
#         full_df = full_df.truncate(after = 299)
#         first_fish = False
#     else:
#         fish = fish_agent(random.randint(MIN_X,MAX_X),
#                           random.randint(MIN_Y,MAX_Y),
#                           random.uniform(0,2*math.pi),
#                           random.gauss(MEAN_SPEED,SD_SPEED),
#                           "fish{x}".format(x=x))
#         fish.move()
#         df = fish.output()
#         df = df.truncate(after = 299)

#         full_df = pd.concat([full_df,df])

# print(full_df)

# full_df = trials[0].return_raw_comp_vals()

# fig = px.scatter(full_df,x="Xs", y="Ys", animation_frame="Time", color = "Num")

# fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

# xmin = MIN_X
# xmax = MAX_X
# ymin = MIN_Y
# ymax = MAX_Y

# fig.update_layout(xaxis_range=[MIN_X,MAX_X])
# fig.update_layout(yaxis_range=[MIN_Y,MAX_Y])

# fig.show()

