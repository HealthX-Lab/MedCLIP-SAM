
#%%
# Plot setting
import pandas as pd
#%%
import matplotlib.pyplot as plt
from matplotlib import cm
color_map = cm.tab10.colors[:4]
plt.grid(True)
plt.style.use('ggplot')
figure_size = (7, 6)
output_folder = 'plots'
#%%
# data source
# multi_class_iou_plot
data_src = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQS7wFdfQany5PcRwLKmUl1EeXosCmNB2Rwnt-M9y1HcLo08kS11d_jSKvNTqmuSyH11z57v_bIorRW/pub?gid=201982815&single=true&output=csv", index_col=0)

#%%
# multi-class IoU plot

#%%
#  = plt.subplots(figsize = figure_size)
ax = data_src.T.plot.bar(rot=0, color=color_map, figsize=figure_size)
plt.ylabel('Mean IoU')
plt.xlabel('Number of classes per image')
plt.savefig('plots/multi_class_iou.jpg')
#%%
# multi class box plot
data_folder = '/home/peijie/phrase_grounding/gscorecam/results/lvis'
bin_list = [[1,2,3], [4,5,6], [7,8,9]]
cam_versions = ['gscorecam', 'gradcam', 'scorecam', 'hilacam']
def load_all(data_folder, cam_versions, bin_list):
    data_raw = {'gscorecam': [], 'gradcam': [], 'scorecam': [], 'hilacam': []}
    for cam_version in cam_versions:
        model = 'ViT-B_32' if 'hila' in cam_version else 'RN50x16'
        for bin in bin_list:
            data = [pd.read_hdf(f'{data_folder}/{model}_{cam_version}_c{i:02d}.hdf5') for i in bin]
            data_raw[cam_version].append(pd.concat(data))
    return data_raw
#%%
#Qi's plot 
raw_data = load_all(data_folder, cam_versions, bin_list)
import numpy as np
fig, ax = plt.subplots(figsize = figure_size)

mark_fig = {'gscorecam': dict(marker='o'), 
            'gradcam': dict(marker='s'),
            'scorecam': dict(marker='X'),
            'hilacam': dict(marker='^')}

box_fig = {'gscorecam': dict(facecolor=color_map[0]), 
            'gradcam': dict(facecolor=color_map[1]),
            'scorecam': dict(facecolor=color_map[2]),
            'hilacam': dict(facecolor=color_map[3])}

pos_adjustment = {'gscorecam': -0.3, 
            'gradcam': -0.1,
            'scorecam': 0.1,
            'hilacam': 0.3}

box_width = 0.15
plot_labels = ['1-3\n(1150)', '4-6\n(2790)', '7-9\n(874)']
axes_boxplot = []
plt.ylim((-0.05, 1.2))
plt.xlim((0.5, 3.5))
for cam in cam_versions:
    x_pos = np.array(range(3))+1+pos_adjustment[cam]
    axes_boxplot.append(ax.boxplot([raw_data[cam][i].max_iou.values for i in range(len(bin_list))], boxprops=box_fig[cam], widths=box_width, positions=x_pos, patch_artist=True, notch=True, flierprops=mark_fig[cam]))
    # for pos, iou_group in zip(x_pos, raw_data[cam]):
    #     plt.text(pos, -0.03, f'{iou_group.max_iou.median():.3f}', horizontalalignment='center', size='small', color=box_fig[cam]['facecolor'], weight='semibold')

# ax.legend([bp['boxes'][0] for bp in axes_boxplot], cam_versions, loc='upper right', bbox_to_anchor=(1.15, 1.00))
legend_names = ['gScoreCAM', 'GradCAM', 'ScoreCAM', 'HilaCAM']
ax.legend([bp['boxes'][0] for bp in axes_boxplot], legend_names, loc='upper right', prop={'size': 15}, bbox_to_anchor=(1.025, 1.025), ncol=2)
plt.xticks(range(1, len(plot_labels)+1), plot_labels, fontsize=18)
plt.yticks(np.linspace(0, 1.0, num=6), fontsize=18)
# ax.set_ylabel()
plt.xlabel("Number of classes per image", fontsize=18)
plt.ylabel("IoU", fontsize=18)
plt.savefig('plots/multi_class_box.pdf', bbox_inches='tight')
# plt.title(f"The IoU for different CAM methods")

# %%
