# Fetch a project.
from neptune.sessions import Session
import seaborn as sns
import matplotlib.pyplot as plt
import config


session = Session(api_token=config.API_KEY)

# project = session.get_projects('4nd4')['4nd4/Vec2UAge']
project = session.get_projects('4nd4')['4nd4/Vec2UAge']
unique_tag = 'C1'
optimizer_filter = 'SGD'

p = (project.get_leaderboard(tag=[unique_tag]))


p['channel_val_mae'] = p['channel_val_mae'].astype(float)
p['channel_mae'] = p['channel_mae'].astype(float)
p['channel_test_mae'] = p['channel_test_mae'].astype(float)

optimizers = ['ADAGRAD', 'ADAM', 'SGD', 'SWA']


def label_opt(row):
    for x in optimizers:

        if x in row['tags']:
            return x


p['optimizer'] = p.apply(lambda row: label_opt(row), axis=1)

if unique_tag == 'B1':
    table_opt = p[["id", "parameter_random_seed", "parameter_learning_rate", "channel_val_mae", "channel_test_mae",
                   "optimizer"]]
    table_opt['parameter_learning_rate'] = table_opt['parameter_learning_rate'].astype(float)
    table_opt = table_opt.drop(['parameter_learning_rate'], axis=1)

else:
    table_opt = p[["id", "parameter_random_seed", "property_param__lr", "channel_val_mae", "channel_test_mae",
                   "optimizer"]]
    table_opt['property_param__lr'] = table_opt['property_param__lr'].astype(float)

filter = table_opt["optimizer"] == optimizer_filter

# filtering data
table_opt.where(filter, inplace=True)

proc_table_opt = table_opt.dropna().drop(['optimizer'], axis=1)

print(proc_table_opt)
proc_table_opt.to_csv('detailed_experiment_{}.csv'.format(unique_tag))


# id
# parameter_random_seed
# property_param__lr
# channel_val_mae

melted_df = p.melt(
    ['id', 'optimizer'],
    ['channel_mae', 'channel_val_mae', 'channel_test_mae'],
    value_name='MAE', var_name='loss'
).groupby(['id', 'loss', 'MAE'], as_index=False).sum()


ax = sns.boxplot(x="optimizer", y="MAE", data=melted_df,
                 hue='loss',
                 order=optimizers,
                 hue_order=['channel_mae', 'channel_val_mae', 'channel_test_mae'],

                 )

ax = sns.stripplot(x="optimizer", y="MAE", data=melted_df,
                   hue='loss',
                   order=optimizers,
                   hue_order=['channel_mae', 'channel_val_mae', 'channel_test_mae'],
                   color='.25',
                   dodge=True,
                   ax=ax,
                   jitter=False,

                   )

# replace labels

new_labels = ['train', 'validation', 'test']

handles, labels = ax.get_legend_handles_labels()

plt.legend(
    handles[0:3], labels[0:3],
)

for i in range(0, len(new_labels)):
    ax.legend_.texts[i].set_text(new_labels[i])

plt.show()

# print counts per optimizer, mean per optimizer and variance

p[['id', 'channel_mae', 'channel_val_mae', 'channel_test_mae', 'optimizer']].groupby(['optimizer']).agg(['min', 'mean', 'std']).to_csv('Vec2UAge_stats.csv')


# print std, count, mean per optimizer for training, test and validation

# print(p['channel_val_mae'].min())
# print(p['channel_val_mae'].mean())
# print(p.var()['channel_val_mae'])
