import matplotlib.pyplot as plt

file_path = '/data/projects/11003765/sate/satte/satellite-te/output/supervised/starlink_500_ISL_spaceTE/models/spaceTE_obj-teal_total_flow_supervised_lr-1e-05_ep-20_sample-200_layers-0_decoder-linear.losses'

losses = []
with open(file_path, 'r') as file:
    for line in file:
        loss = float(line.strip())
        losses.append(loss)

plt.figure(figsize=(10, 6))
plt.plot(losses, label='Training Loss')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()