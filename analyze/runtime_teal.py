import pandas as pd

file_iridium = '/data/projects/11003765/sate/satte/satellite-te/output/teal/iridium_IridiumDataSet14day20sec_Int15_teal/test_logs/teal_topo-1_tsz-None_vr-0.2_lr-0.0001_ep-3_bsz-1_layer-6_sample-5_rho-1.0_step-5.csv'

file_176 = '/data/projects/11003765/sate/satte/satellite-te/output/teal/starlink_176_ISL_teal_teal/test_logs/teal_topo-1_tsz-None_vr-0.2_lr-0.0001_ep-3_bsz-1_layer-6_sample-5_rho-1.0_step-5.csv'

file_500 = '/data/projects/11003765/sate/satte/satellite-te/output/teal/starlink_500_ISL_teal_teal/test_logs/teal_topo-1_tsz-None_vr-0.2_lr-0.0001_ep-3_bsz-1_layer-6_sample-5_rho-1.0_step-5.csv'

file_528 = '/data/projects/11003765/sate/satte/satellite-te/output/teal/starlink_528_ISL_teal_teal/test_logs/teal_topo-1_tsz-None_vr-0.2_lr-0.0001_ep-3_bsz-1_layer-6_sample-5_rho-1.0_step-5.csv'

df_iridium = pd.read_csv(file_iridium)
df_176 = pd.read_csv(file_176)
df_500 = pd.read_csv(file_500)
df_528 = pd.read_csv(file_528)

mean_runtime_iridium = df_iridium['runtime'].mean()
mean_runtime_176 = df_176['runtime'].mean()
mean_runtime_500 = df_500['runtime'].mean()
mean_runtime_528 = df_528['runtime'].mean()

print(mean_runtime_iridium, mean_runtime_176, mean_runtime_500, mean_runtime_528)
