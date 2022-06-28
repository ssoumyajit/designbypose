import sys
from moviepy.editor import *
import moviepy.editor as mpy
# import argparse

'''
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputvideo", required=True, help="path to the input video")
ap.add_argument("-v", "--video", required=True, help="path to the input video")
ap.add_argument("-r", "--result", required=True, help="path to the input video")
'''


#args = vars(ap.parse_args)
# extract audio from a given video
videoin = mpy.VideoFileClip("/Users/river/Downloads/mediain/tim.mp4")
audio = videoin.audio
audio.write_audiofile("audio.mp3")

video = mpy.VideoFileClip("/Users/river/Downloads/mediaout/timmix.mp4")
video.write_videofile("timf1.mov", codec='libx264', audio_codec='aac', audio='audio.mp3', remove_temp=True) 
