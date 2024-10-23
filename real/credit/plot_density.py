# #%%

# def plot_density(y0, y1, label0='y0', label1='y1'):
#     # if isinstance(y0, torch.Tensor): y0 = y0.detach().numpy()
#     # if isinstance(y1, torch.Tensor): y1 = y0.detach().numpy()
#     cur_ls = [y0.values, y1.values]
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.hist(cur_ls, density=True, bins=20, alpha = 0.5, histtype='bar', 
#             label = [label0, label0], stacked=True)
#     ax.legend()
#     ax.set_title('different sample sizes')
#     plt.show()

#     return fig


# #%%

# import numpy as np
# import matplotlib.pyplot as plt

# np.random.seed(19680801)

# n_bins = 10
# x = np.random.randn(1000, 3)

# fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)

# colors = ['red', 'tan', 'lime']
# ax0.hist(x, n_bins, density=True, histtype='bar', color=colors, label=colors)
# ax0.legend(prop={'size': 10})
# ax0.set_title('bars with legend')

# ax1.hist(x, n_bins, density=True, histtype='bar', stacked=True)
# ax1.set_title('stacked bar')

# ax2.hist(x, n_bins, histtype='step', stacked=True, fill=False)
# ax2.set_title('stack step (unfilled)')

# # Make a multiple-histogram of data-sets with different length.
# x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
# ax3.hist(x_multi, n_bins, histtype='bar')
# ax3.set_title('different sample sizes')

# fig.tight_layout()
# plt.show()


# #%%
# k = 8
# n_bins = np.arange(-0.5, 2.5)
# cur_ls = [Full_y0.iloc[k].values, Full_y1.iloc[k].values, Unaware_y0.iloc[k].values, Unaware_y1.iloc[k].values]
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.hist(cur_ls, density=True, bins=n_bins, alpha = 0.5, 
#         histtype='bar', label = ['y0', 'y1'], stacked=True,
#         )
# ax.legend()
# ax.set_title(f'{k} graph\'s credit density plot')
# plt.xticks([0,1],['Full','Full'],fontsize=10)
# plt.savefig(dir+f'/density_{k}_graph.png')
# plt.show()

# #%%

# plt.bar([0,1],[sum(Full_y0.iloc[k]==0), sum(Full_y0.iloc[k]==1)], label='y0')

# #%%

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# k = 1

# df = pd.DataFrame(
#     {
#         'y0_0': [sum(Full_y0.iloc[k]==0), sum(Unaware_y0.iloc[k]==0), sum(Fair10_y0.iloc[k]==0), sum(Fair150_y0.iloc[k]==0)],
#         'y0_1': [sum(Full_y0.iloc[k]==1), sum(Unaware_y0.iloc[k]==1), sum(Fair10_y0.iloc[k]==1), sum(Fair150_y0.iloc[k]==1)],
#         'y1_0': [sum(Full_y1.iloc[k]==0), sum(Unaware_y1.iloc[k]==0), sum(Fair10_y1.iloc[k]==0), sum(Fair150_y1.iloc[k]==0)],
#         'y1_1': [sum(Full_y1.iloc[k]==1), sum(Unaware_y1.iloc[k]==1), sum(Fair10_y1.iloc[k]==1), sum(Fair150_y1.iloc[k]==1)],
#         'label': ['Full', 'Unaware', 'Fair10', 'Fair150']
#     })


# x = np.arange(0,len(df['label'])*2,2)
# width=0.5
# x1 = x-width/2
# x2 = x+width/2
# y1 = df['y0_0']
# y2 = df['y0_1']

# # 绘制分组柱状图

# plt.bar(x1,y1,width=0.5,label='北京',color='#f9766e',edgecolor='k',zorder=2)
# plt.bar(x2,y2,width=0.5,label='广州',color='#00bfc4',edgecolor='k',zorder=2)

# # 添加x,y轴名称、图例和网格线
# plt.xlabel('2022年',fontsize=11)
# plt.ylabel('AQI',fontsize=11)
# plt.legend(frameon=False)
# plt.grid(ls='--',alpha=0.8)

# # 修改x刻度标签为对应日期
# plt.xticks(x,df['label'],fontsize=10)
# plt.tick_params(axis='x',length=0)

# plt.tight_layout()
# plt.savefig('bar2.png',dpi=600)
# plt.show()


# #%%
# import seaborn as sns

# # plt.style.use('ggplot')
# plt.rcParams['axes.unicode_minus'] = False

# k = 8
# n_bins = 20 # np.arange(-0.5, 2.5)
# # cur_ls = [Full_y0.iloc[k].values, Full_y1.iloc[k].values, Unaware_y0.iloc[k].values, Unaware_y1.iloc[k].values]
# cur_ls = [Full_y0.iloc[k], Full_y1.iloc[k], Unaware_y0.iloc[k].values, Unaware_y1.iloc[k].values]

# sns.histplot(pd.DataFrame(cur_ls[0]), bins=n_bins, kde=False, color='steelblue', label = 'y0') 
# sns.histplot(pd.DataFrame(cur_ls[1]), bins=n_bins, kde=False, color='orange', label = 'y1')
# plt.title(f'{k} graph\'s credit density plot')
# plt.xlabel('credit')
# plt.ylabel('count')
# plt.legend()
# # plt.xticks([0,1],['Full','Full'],fontsize=10)
# # plt.savefig(dir+f'/density_{k}_graph.png')
# plt.show()


#######################################################################
#                           Start from here                           #
#######################################################################

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read in the data
dir = 'D:\\universityWorks\\thirdYear\\Spring\\Aoqi Zuo\\0508\\credit/results/saves/05100100/y_pred'

Full_y0 = pd.read_csv(dir+"/Full_y0.csv", header = None)
Full_y1 = pd.read_csv(dir+"/Full_y1.csv", header = None)
sample_size = Full_y0.shape[1]
Unaware_y0 = pd.read_csv(dir+"/Unaware_y0.csv", header = None)
Unaware_y1 = pd.read_csv(dir+"/Unaware_y1.csv", header = None)
Fair10_y0 = pd.read_csv(dir+"/e-IFair_10_y0.csv", header = None)
Fair10_y1 = pd.read_csv(dir+"/e-IFair_10_y1.csv", header = None)
Fair150_y0 = pd.read_csv(dir+"/e-IFair_150_y0.csv", header = None)
Fair150_y1 = pd.read_csv(dir+"/e-IFair_150_y1.csv", header = None)


#%%
k = 4
width = 0.9
ALPHA = 0.5

# Assuming you have two binary variables: variable1 and variable2
# prefix = '         No       Yes\n          '
# postfix = ''
# categories_labels = [prefix+'Full'+postfix, '', ' ', prefix+'Unaware'+postfix, '', ' ', prefix+'Fair10'+postfix, '', ' ', prefix+'Fair150'+postfix, '']

prefix = '             0     1'
postfix = ''
gap = '     '
# categories = ['Full_y0', 'Full_y1', gap, gap, 'Unaware_y0', 'Unaware_y1', gap, gap, 'Fair10_y0', 'Fair10_y1', gap, gap, 'Fair150_y0', 'Fair150_y1']

# categories_labels = [prefix+gap+postfix, '', ' ', ' ', prefix+gap+postfix, '', ' ', ' ', prefix+gap+postfix, '', ' ', ' ', prefix+gap+postfix, '']
# counts_variable0 = [sum(Full_y0.iloc[k]==0)/len(Full_y0.iloc[k]), sum(Full_y0.iloc[k]==1)/len(Full_y0.iloc[k]), 0, 0,
#                     sum(Unaware_y0.iloc[k]==0)/len(Unaware_y0.iloc[k]), sum(Unaware_y0.iloc[k]==1)/len(Unaware_y0.iloc[k]), 0, 0,
#                     sum(Fair10_y0.iloc[k]==0)/len(Fair10_y0.iloc[k]), sum(Fair10_y0.iloc[k]==1)/len(Fair10_y0.iloc[k]), 0, 0,
#                     sum(Fair150_y0.iloc[k]==0)/len(Fair150_y0.iloc[k]), sum(Fair150_y0.iloc[k]==1)/len(Fair150_y0.iloc[k])]
# counts_variable1 = [sum(Full_y1.iloc[k]==0)/len(Full_y1.iloc[k]), sum(Full_y1.iloc[k]==1)/len(Full_y1.iloc[k]), 0, 0,
#                     sum(Unaware_y1.iloc[k]==0)/len(Unaware_y1.iloc[k]), sum(Unaware_y1.iloc[k]==1)/len(Unaware_y1.iloc[k]), 0, 0,
#                     sum(Fair10_y1.iloc[k]==0)/len(Fair10_y1.iloc[k]), sum(Fair10_y1.iloc[k]==1)/len(Fair10_y1.iloc[k]), 0, 0,
#                     sum(Fair150_y1.iloc[k]==0)/len(Fair150_y1.iloc[k]), sum(Fair150_y1.iloc[k]==1)/len(Fair150_y1.iloc[k])]

categories = ['Full_y0', 'Full_y1', gap, 'Unaware_y0', 'Unaware_y1', gap, gap, 'Fair10_y0', 'Fair10_y1', gap, 'Fair150_y0', 'Fair150_y1']

categories_labels = [prefix+gap+postfix, '', ' ', prefix+gap+postfix, '', ' ', ' ', prefix+gap+postfix, '', ' ', prefix+gap+postfix, '']
counts_variable0 = [sum(Full_y0.iloc[k]==0)/len(Full_y0.iloc[k]), sum(Full_y0.iloc[k]==1)/len(Full_y0.iloc[k]), 0, 
                    sum(Unaware_y0.iloc[k]==0)/len(Unaware_y0.iloc[k]), sum(Unaware_y0.iloc[k]==1)/len(Unaware_y0.iloc[k]), 0, 0,
                    sum(Fair10_y0.iloc[k]==0)/len(Fair10_y0.iloc[k]), sum(Fair10_y0.iloc[k]==1)/len(Fair10_y0.iloc[k]), 0, 
                    sum(Fair150_y0.iloc[k]==0)/len(Fair150_y0.iloc[k]), sum(Fair150_y0.iloc[k]==1)/len(Fair150_y0.iloc[k])]
counts_variable1 = [sum(Full_y1.iloc[k]==0)/len(Full_y1.iloc[k]), sum(Full_y1.iloc[k]==1)/len(Full_y1.iloc[k]), 0, 
                    sum(Unaware_y1.iloc[k]==0)/len(Unaware_y1.iloc[k]), sum(Unaware_y1.iloc[k]==1)/len(Unaware_y1.iloc[k]), 0, 0,
                    sum(Fair10_y1.iloc[k]==0)/len(Fair10_y1.iloc[k]), sum(Fair10_y1.iloc[k]==1)/len(Fair10_y1.iloc[k]), 0, 
                    sum(Fair150_y1.iloc[k]==0)/len(Fair150_y1.iloc[k]), sum(Fair150_y1.iloc[k]==1)/len(Fair150_y1.iloc[k])]


# Plot the bar chart
x = np.array(list(range(len(categories))))
fig, ax = plt.subplots(figsize=(10,6))

rects1 = ax.bar(x, counts_variable0, width, label=r'$\rm\hat{Y}_{A\leftarrow a}$', alpha=ALPHA, color='#F4A582')
rects2 = ax.bar(x, counts_variable1, width, label=r"$\rm\hat{Y}_{A\leftarrow a'}$", alpha=ALPHA, color='#92C5DE')

# Customize the plot
ax.set_xticks(x)
ax.set_xticklabels(categories_labels)
ax.set_ylabel('Probability', fontsize=40, labelpad = 30)
ax.set_xlabel('  ' .join([r'$\rm\hat{Y}$(Full)', r'$\rm\hat{Y}$(Unaware)', 
                        #    r'$\rm\hat{Y}$($\epsilon$-IFair, $\lambda=10$)',
                        #    r'$\rm\hat{Y}$($\epsilon$-IFair, $\lambda=150$)',
                           r'$\rm\hat{Y}$($\epsilon$-IFair)',
                           r'$\rm\hat{Y}$($\epsilon$-IFair)',
                           ])
                , fontsize=30, labelpad = 5)

# ax.set_title(f'Credit dataset {k} graph\'s density plot')
ax.set_title(f' ')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, fontsize=42, framealpha = 0)
plt.ylim(0,1)
# plt.xlim(-3,15)

# plt.legend(fontsize=24)  # Set the font size for the legend
plt.xticks(fontsize=28)  # Set the font size for the x-axis tick labels
plt.yticks(fontsize=30)  # Set the font size for the y-axis tick labels


# Show the plot
# plt.savefig(dir+f'/Credit_density_{k}_graph.pdf',dpi=600, bbox_inches='tight')
plt.savefig(dir+f'/Credit_density.pdf',dpi=600, bbox_inches='tight')
plt.show()
#%%
