import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import argparse



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('plot_csv.py')
    add_arg = parser.add_argument
    add_arg('--csv_path', default='./')
    add_arg('-y', '--yaxis', choices=['flops', 'lus_read_bytes', 'lus_read_usec'])
    add_arg('-o', '--output', default='combined.png')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--ranks', type=int, default=1)
    add_arg('--max_x', type=int, default=-1)
    return parser.parse_args()


args = parse_args()


def read_datafile(file_name):
    # the skiprows keyword is for heading, but I don't know if trailing lines
    # can be specified
    data = np.loadtxt(file_name, delimiter=',', skiprows=1)
    return data

#csv_name_list=['cpu_freq','freemem','node_power','req_bytes_recv','req_bytes_sent','lus_read_calls']
import matplotlib.patches as mpatches 
csv_name_list=['flops','read_bytes']
color_list=['red','blue']
rank_list=[i for i in range(0,args.ranks)]
#csv_name='freemem-0'
#data = read_datafile('e:\dir1\datafile.csv')
fig = plt.figure()
fig2 = plt.figure()
ax0 = fig2.add_subplot(111)
#ax0.set_title(args.output + ": Per-Worker FLOPs and Read-Bytes")
for i in range(len(csv_name_list)):
    subplotnum=211+i
    ax0.set_title(args.output + ": Per-Worker " + str(csv_name_list[i]).upper())
    ax1 = fig.add_subplot(subplotnum)
    title=csv_name_list[i] + ' vs Time '
    ax1.set_title(title)
    ax1.set_xlabel('Time')
    ax0.set_xlabel('Time')
    #leg = ax1.legend()
    handles_list = []
    for j in range(len(rank_list)):
        csv_name=csv_name_list[i] + '-' + str(rank_list[j])
        csv_path=args.csv_path + csv_name + '.csv'
	    #csv_path='/lus/scratch/jbalma/coral2/keras_imagenet_resnet50_noio_profiled/' + csv_name + '.csv'
        #csv_path='/lus/scratch/jbalma/temp/heptrkx-gnn-hitgraphs_big_000-np16_16nodes_1ppn_34omp_loadbal_profile/' + csv_name + '.csv'
        #csv_path='/lus/scratch/jbalma/temp/heptrkx-gnn-hitgraphs_big_000-np16_16nodes_1ppn_34omp_shuffle_profile/' + csv_name + '.csv'
        #csv_path='/lus/scratch/jbalma/temp/heptrkx-gnn-hitgraphs_big_000-np16_16nodes_1ppn_68omp_orig/' + csv_name + '.csv'
        data = np.genfromtxt(csv_path, delimiter=',', skip_header=1, skip_footer=0, names=['t', 'f'])
        #print("data.t=",data['t'])
        #print("data.f=",data['f'])
        #x = data.t
        #y = data.f
        #ax1.set(xlim=(200, 5000), ylim=(1E6, 1E10))
        lower_lim=1
        if(csv_name_list[i]=='flops'):
            lower_lim = 1E8 
        elif(csv_name_list[i]=='read_bytes'):
            lower_lim = 1

        ax1.set(ylim=(lower_lim, 1E11))
        ax0.set(ylim=(lower_lim, 1E11))
        #plt.yscale("log")
        #fig = plt.figure()

        #ax1 = fig.add_subplot(111)
        #	title=csv_name + 'vs Time '
        #	ax1.set_title("Mains power stability") 
        #	ax1.set_xlabel('Time')
        #	ax1.set_ylabel(csv_name)
    	#ax1.set_alpha(1.0/float(rank_list[j]+0.1))
        ax1.set_prop_cycle("alpha", str(1.0/(j+1.0)))
        ax0.set_prop_cycle("alpha", str(1.0/(j+1.0)))
        patch = mpatches.Patch(color=color_list[i],label=csv_name)
        if(j==0):
            handles_list.append(patch)
        #plt.legend(handles=[red_patch])
        plt.yscale("log")
        #ax1.plot(data['t'], data['f'], color=color_list[i], label=csv_name)
        ax1.plot(data['t'][0:args.max_x], data['f'][0:args.max_x], color=color_list[i], alpha=0.1, label=csv_name)
        ax0.plot(data['t'][0:args.max_x], data['f'][0:args.max_x], color=color_list[i], alpha=0.1, label=csv_name)

    plt_name=csv_name_list[i]+args.output 
    #plt_name=args.output
    plt.legend(handles=handles_list)
    plt.savefig(plt_name)
    fig.clear()
plt.savefig("combined.png")
fig2.clear()


print("Done.")
#plt.show()
