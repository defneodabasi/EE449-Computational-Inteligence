# -*- coding: utf-8 -*-
"""
Created on Wed May 22 23:35:30 2024

@author: defne
"""

import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def animation_gif(directory,starting_episode,ending_episode):
    #read all the .png files in directory called `steps`
    files = sorted(glob.glob(os.path.join(directory, '*.png')))

    image_array = []
    for my_image in files:
        image = Image.open(my_image)
        image_array.append(image)
    # Create the figure and axes objects
    fig, ax = plt.subplots()
    
    # Set the initial image
    im = ax.imshow(image_array[0], animated=True)
    
    def update(i):
        im.set_array(image_array[i])
        return im, 
    
    # Create the animation object
    animation_fig = animation.FuncAnimation(fig, update, frames=len(image_array), interval=200, blit=True,repeat_delay=10)
    
    # Show the animation
    plt.show()
    
    file_name = f"animation_output_{starting_episode}_{ending_episode}"
    file_path = 'C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE449\\HW3\\animation_gifs\\animation_output_QL_' + file_name + '.gif'
    animation_fig.save(file_path)