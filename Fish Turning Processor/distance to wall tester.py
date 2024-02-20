import matplotlib.pyplot as plt
import numpy as np


wall_lines = np.array([[[0,2],[6,2]],
                       [[6,2],[6,0]],
                       [[6,0],[0,0]],
                       [[0,0],[0,2]]])

def closest_right_distance_to_wall(headX, headY, midlineX, midlineY):

    #find all the distances from every wall
    distances = np.full([4], np.nan)
    percents = np.full([4], np.nan)

    for i,wall in enumerate(wall_lines):

        #See math document from Eric 
        #If you're not me, email me I guess?
        #But this gives k and b 
        # where k is the mutliplier of vector n that makes it intersect with vector w (wall)
        # and where b is the mutliplier of vector w that makes it intersect with vector n (right perpendicular to fish)
        nX = (headY - midlineY)
        nY = -1*(headX - midlineX)

        a = np.array([[wall[0][0] - headX],
                      [wall[0][1] - headY]])

        b = np.array([[nX , wall[0][0] - wall[1][0]],
                      [nY , wall[0][1] - wall[1][1]]])

        invB = np.linalg.inv(b)
        answer = np.matmul(invB,a)

        #This gives us the distance on the right the fish is from the wall
        #And the percent along the wall that line intersects with
        distances[i] = answer[0][0] * np.sqrt(nX**2 + nY**2)
        percents[i] = answer[1][0]

    #Then we only get the none negative distance snad percents (the real ones)
    real_distances = (distances * (distances >= 0) * (percents >= 0))

    #Then find the closest distance o those, and the percent that goes along with it
    closest_real_dist = np.nanmin(np.where(real_distances == 0, np.nan, real_distances))
    closest_real_percent = percents[np.where(distances == closest_real_dist)[0][0]]

    return(closest_real_dist, closest_real_percent)

print(closest_right_distance_to_wall(1.5,1.5,2,1))