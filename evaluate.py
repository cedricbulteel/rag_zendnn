import pandas as pd
import glob

csv_files = glob.glob("rag_benchmark_*.csv")

dfs = [pd.read_csv(file) for file in csv_files]

dfs_torch = [df[df['backend'] == 'torch'] for df in dfs]
dfs_torch = [df.drop(columns=['backend']) for df in dfs_torch]
dfs_zentorch = [df[df['backend'] == 'zentorch'] for df in dfs]
dfs_zentorch = [df.drop(columns=['backend']) for df in dfs_zentorch]

mean_df_torch = pd.concat(dfs_torch).groupby(level=0).mean()
mean_df_zentorch = pd.concat(dfs_zentorch).groupby(level=0).mean()
print("Mean (default):")
print(mean_df_torch)
print("Mean (zentorch)")
print(mean_df_zentorch)

df_relative_diff = (mean_df_torch.set_index("batch_size") - mean_df_zentorch.set_index("batch_size")) / mean_df_torch.set_index("batch_size")*100

# Reset index and remove 'Unnamed: 0'
df_relative_diff.reset_index(inplace=True)

print("Relative difference (%)")
print(df_relative_diff)

mean_df_torch.to_csv('torch_mean.csv')
mean_df_zentorch.to_csv('zentorch_mean.csv')
df_relative_diff.to_csv('difference.csv')
