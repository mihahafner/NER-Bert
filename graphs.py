import shutil
import os
import csv
from matplotlib import pyplot as plt

data = []
for fileName in os.listdir('models'):
    if fileName.endswith('csv'):

        info = fileName.replace('.csv', '').split('_')
        with open(f'models/{fileName}', 'r') as file:
            rows = []
            for line in csv.DictReader(file):
                rows.append({key: float(value) for key, value in line.items()})
            print(rows)

            data.append({
                'file': fileName,
                'optimizer': info[0],
                'learn_rate': 10**int(info[1].replace('LR1', '')),
                'batch_size': int(info[2].replace('BS', '')),
                'rows': rows
            })
try:
    shutil.rmtree('graphs')
except:
    pass
os.mkdir('graphs')


# All model in seperate folder
for model in data:
    opti = model['optimizer']
    lr = model['learn_rate']
    batch = model['batch_size']
    rows = model['rows']
    file_name = model['file']
    x = [row['epoch'] for row in rows]
    val_loss = [row['val_loss'] for row in rows]
    train_loss = [row['train_loss'] for row in rows]
    val_acc= [row['val_acc'] for row in rows]

    path = f'graphs/{file_name.replace(".csv", "")}'
    os.mkdir(path)

    plt.title(f'Optimizer: {opti}, Learn rate: {lr}, Batch size: {batch}')
    plt.plot(x, val_loss)
    plt.plot(x, train_loss)
    plt.legend(['Validation loss', 'Train loss'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f'{path}/loss.png')
    plt.clf()
    plt.cla()

    plt.plot(x, val_acc)
    plt.title(f'Optimizer: {opti}, Learn rate: {lr}, Batch size: {batch}')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(['Accuracy'])
    plt.savefig(f'{path}/acc.png')
    plt.clf()
    plt.cla()

optimizers_color = {
    'Adam': 'red',
    'Adadelta': 'blue',
}

# All optimizer in the same folder
legend = []
scatter_x = []
scatter_y = []
scatter_area = []
scatter_model = []
for model in data:
    opti = model['optimizer']
    lr = model['learn_rate']
    batch = model['batch_size']
    rows = model['rows']
    file_name = model['file']
    x = [row['epoch'] for row in rows]
    val_loss = [row['val_loss'] for row in rows]
    train_loss = [row['train_loss'] for row in rows]
    val_acc= [row['val_acc'] for row in rows]

    scatter_x.append(batch)
    scatter_y.append(lr)
    scatter_model.append(opti)
    scatter_area.append(val_loss[-1])

    plt.plot(x, val_loss)
    legend.append(f'{opti} LR{lr:.2E} batch {batch}')

path = f'graphs/all'
os.mkdir(path)

plt.title(f'All models validation loss functions')
plt.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout(rect=[0,0, 1, 1])
plt.savefig(f'{path}/validation_loss.png')
plt.clf()
plt.cla()

with open(f'{path}/learnRate_batch.csv', 'w') as file:
    csvFile = csv.DictWriter(file, fieldnames=['model', 'learn_rate', 'batch_size', 'val_loss'])
    csvFile.writeheader()
    for i in range(len(scatter_y)):
        csvFile.writerow({
            'model': scatter_model[i],
            'learn_rate': scatter_y[i],
            'batch_size': scatter_x[i],
            'val_loss': scatter_area[i],
        })

