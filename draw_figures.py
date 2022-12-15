import matplotlib.pyplot as plt
import numpy as np

def draw_fig(acc_list, batch, learning_rate):
  x_value = list(range(1, len(acc_list) + 1))
  y_value = acc_list
  plt.style.use = ('seaborn')
  fig,ax = plt.subplots()
  plt.xlim((-1, len(acc_list) + 5))
  plt.ylim((0, 1))
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_position(('data', 0))
  ax.spines['left'].set_position(('data', 0))
  ax.tick_params(axis='both', which='major', direction='in', labelsize=10)
  ax.set_title(f"BATCH_SIZE: {batch}", fontsize=16)
  ax.set_xlabel("EPOCH", fontsize=11, loc='right')
  ax.set_ylabel("ACCURACY",fontsize=11, loc='top')
  ax.scatter(x_value, y_value, s=4)
  line1, = ax.plot(x_value, y_value, linewidth=2, linestyle='-')
  fig.legend([line1], [f'BATCH = {batch}'], loc='center', bbox_to_anchor=(0.75, 0.18))
  fig.savefig(f'./figures/lr{learning_rate}_batch_{batch}.jpg')
  # plt.show()


if __name__ == '__main__':
  all_batch_acc = []
  for i in range(1, 26):
    file=open(f'./results/batch{i}_eval.txt')
    acc_list = []
    for line in file.readlines():
      curLine = line.strip().split('/')
      # print(curLine[1])
      acc_list.append(float(curLine[1]))
    all_batch_acc.append(acc_list)
    # draw_fig(acc_list, i)

  idx = 0
  for item in all_batch_acc:
    print()
    if len(item) > 500:
      all_batch_acc[idx] = item[0: 501]
    draw_fig(all_batch_acc[idx], idx + 1)
    idx += 1


