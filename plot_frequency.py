#!/usr/bin/python

import argparse

from rosbag import Bag

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rospy

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot frequency of the given topics from an input bag file')
    parser.add_argument("-b", "--bag", required=True, nargs=1, help="input bag file")
    parser.add_argument("-t", "--topics", required=True, nargs="*", help="list of topics to plot")
    args = parser.parse_args()

  
    # Open bag file
    with Bag(args.bag[0], 'r') as bag:

        first_iteration = {t:True for t in args.topics}
        periods = {t:[] for t in args.topics}
        first_times = {t:0 for t in args.topics}
        times = {t:[] for t in args.topics}
        prev_time = {t:rospy.Time() for t in args.topics} 

        # Get topics frequency
        for topic, msg, t in bag.read_messages(topics=args.topics):
            if first_iteration[topic]:
                first_times[topic] = msg.header.stamp.to_sec()
                prev_time[topic] = msg.header.stamp
                first_iteration[topic] = False
            else:
                times[topic].append( msg.header.stamp.to_sec() - first_times[topic] )
                periods[topic].append( (msg.header.stamp - prev_time[topic]).to_sec() )
                prev_time[topic] = msg.header.stamp

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        colors = []
        patches = []
        for t in args.topics:
            p = ax.plot(times[t], periods[t])
            colors.append(p[0].get_color())
            patches.append( mpatches.Patch(color=p[0].get_color(), label=t) )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Sensor period (s)")
            ax.grid(True)
        
            ax2.plot(times[t], [0] * len(times[t]), linestyle='--', marker='o', color=p[0].get_color())
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Received messages")
            ax2.grid(True)

        ax.legend(handles=patches)
        ax2.legend(handles=patches)
        plt.show()
            
