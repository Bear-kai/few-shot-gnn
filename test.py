import numpy as np
from utils import io_utils
from data import generator
from torch.autograd import Variable
import torch
import torch.nn as nn
from tqdm import tqdm


def test_one_shot(args, model, test_samples=5000, partition='test'):
    io = io_utils.IOStream('checkpoints/' + args.exp_name + '/run.log')

    io.cprint('\n**** TESTING WITH %s ***' % (partition,))

    loader = generator.Generator(args.dataset_root, args, partition=partition, dataset=args.dataset)

    [enc_nn, metric_nn, softmax_module] = model
    enc_nn.eval()
    metric_nn.eval()
    correct = 0
    total = 0
    iterations = int(test_samples/args.batch_size_test)
    for i in range(iterations):
        data = loader.get_task_batch(batch_size=args.batch_size_test, n_way=args.test_N_way,
                                     num_shots=args.test_N_shots, unlabeled_extra=args.unlabeled_extra)
        [x, labels_x_cpu, _, _, xi_s, labels_yi_cpu, oracles_yi, hidden_labels] = data

        if args.cuda:
            xi_s = [batch_xi.cuda() for batch_xi in xi_s]
            labels_yi = [label_yi.cuda() for label_yi in labels_yi_cpu]
            oracles_yi = [oracle_yi.cuda() for oracle_yi in oracles_yi]
            hidden_labels = hidden_labels.cuda()
            x = x.cuda()
        else:
            labels_yi = labels_yi_cpu

        xi_s = [Variable(batch_xi) for batch_xi in xi_s]
        labels_yi = [Variable(label_yi) for label_yi in labels_yi]
        oracles_yi = [Variable(oracle_yi) for oracle_yi in oracles_yi]
        hidden_labels = Variable(hidden_labels)
        x = Variable(x)

        # Compute embedding from x and xi_s
        z = enc_nn(x)[-1]
        zi_s = [enc_nn(batch_xi)[-1] for batch_xi in xi_s]

        # Compute metric from embeddings
        output, out_logits = metric_nn(inputs=[z, zi_s, labels_yi, oracles_yi, hidden_labels])
        output = out_logits
        y_pred = softmax_module.forward(output)
        y_pred = y_pred.data.cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        labels_x_cpu = labels_x_cpu.numpy()
        labels_x_cpu = np.argmax(labels_x_cpu, axis=1)

        for row_i in range(y_pred.shape[0]):
            if y_pred[row_i] == labels_x_cpu[row_i]:
                correct += 1
            total += 1

        if (i+1) % 100 == 0:
            io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0*correct/total))

    io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0*correct/total))
    io.cprint('*** TEST FINISHED ***\n'.format(correct, total, 100.0 * correct / total))
    enc_nn.train()
    metric_nn.train()

    return 100.0 * correct / total



def test_all_symbols(args, model, test_samples=5000, partition='test'):
    io = io_utils.IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint('\n**** TESTING ALL SYMBOLS WITH %s ***' % (partition,))

    loader = generator.Generator(args.dataset_root, args, partition=partition, dataset=args.dataset)

    [enc_nn, metric_nn, softmax_module] = model
    enc_nn.eval()
    metric_nn.eval()
    correct = 0
    total = 0
    iterations = int(test_samples/args.batch_size_test)
    for n_iter in tqdm(range(iterations)):

        # The N_way testing must be the same as the training N_way in training.
        # The solution to classify a new sample that we do not know what can be from all
        # the classes is to select compute an embedding for all the classes in the dataset.
        # and then compute the metrics with the same N_way as in the traininig, always keeping in the
        # n_way batch the class with highest confidence from the previous batch.

        max_classes = len(loader.data)

        data = loader.get_test_sample(batch_size=args.batch_size_test, n_way=args.test_N_way, num_shots=args.test_N_shots)
        [x, labels_x_cpu] = data
        if args.cuda:
            x = x.cuda()
        x = Variable(x)
        # Compute embedding x
        z = enc_nn(x)[-1]

        hist_winner_classes = np.zeros((args.batch_size_test,max_classes))
        first_batch_lowest_scores = []
        classes_to_do = np.arange(max_classes)[np.newaxis,:].repeat(args.batch_size_test,axis=0)
        isFinished = False
        while not isFinished:

            classes_to_do_positive = np.vstack([i[i>=0] for i in classes_to_do])
            if classes_to_do_positive.shape[1] < args.test_N_way:
                isFinished = True
                add_classes = args.test_N_way - classes_to_do_positive.shape[1]
                arr_add_classes_tmp = np.array(first_batch_lowest_scores)[:,np.newaxis].repeat(add_classes, axis=1)
                classes_to_do_positive = np.hstack((classes_to_do_positive, arr_add_classes_tmp))
            selected_classes = np.vstack([np.random.choice(i,args.test_N_way,replace=False) for i in classes_to_do_positive])

            # condition : np.sum(selected_classes.astype(np.int)[0,:] == labels_x_cpu.int().sum(dim=1)[0]) == 1

            data = loader.get_test_batch(fixed_classes=selected_classes,batch_size=args.batch_size_test, n_way=args.test_N_way,
                                     num_shots=args.test_N_shots, unlabeled_extra=args.unlabeled_extra)
            [xi_s, labels_yi_cpu, oracles_yi, hidden_labels] = data

            if args.cuda:
                xi_s = [batch_xi.cuda() for batch_xi in xi_s]
                labels_yi = [label_yi.cuda() for label_yi in labels_yi_cpu]
                oracles_yi = [oracle_yi.cuda() for oracle_yi in oracles_yi]
                hidden_labels = hidden_labels.cuda()
                x = x.cuda()
            else:
                labels_yi = labels_yi_cpu

            xi_s = [Variable(batch_xi) for batch_xi in xi_s]
            labels_yi = [Variable(label_yi) for label_yi in labels_yi]
            oracles_yi = [Variable(oracle_yi) for oracle_yi in oracles_yi]
            hidden_labels = Variable(hidden_labels)

            # Compute embedding xi_s
            zi_s = [enc_nn(batch_xi)[-1] for batch_xi in xi_s]

            # Compute metric from embeddings
            output, out_logits = metric_nn(inputs=[z, zi_s, labels_yi, oracles_yi, hidden_labels])
            output = out_logits
            y_pred = softmax_module.forward(output)
            y_pred = y_pred.data.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)

            for i, sel_classes in enumerate(selected_classes):
                hist_winner_classes[i][sel_classes[y_pred[i]]] +=1

            #for i, sel_classes in enumerate(selected_classes):
            #    print ("%d, %d: %d" % (i, sel_classes[y_pred[i]], hist_winner_classes[i][sel_classes[y_pred[i]]]))

            #print("+++++++++++++++++++++")
            #for i,class_ in enumerate(selected_classes[0, :]):
            #    print("%d: %d" % (class_, hist_winner_classes[0][class_]))
            #print("+++++++++++++++++++++")

            # In the first n-way find the labels at each index of the batch with lowest score and keep the indexes
            # to add them in the finel n-way if there is not enough labels to form a batch.
            if first_batch_lowest_scores == []:
                for i in np.arange(args.batch_size_test):
                    lowest_score = np.arange(args.test_N_way)
                    lowest_score = np.delete(lowest_score, y_pred[i])
                    # TODO: instead the first one, select the one with lowest score.
                    lowest_score = lowest_score[0]
                    first_batch_lowest_scores.append(selected_classes[i][lowest_score])

            # Remove the classes in the batch that didn't have the highest score
            for i in np.arange(args.batch_size_test):
                lowest_score = np.arange(args.test_N_way)
                lowest_score = np.delete(lowest_score, y_pred[i])
                classes_lowest_score = selected_classes[i][lowest_score]
                classes_to_do[i][classes_lowest_score] = -1


        #print('Expected class in 0: %d. Predicted class in 0: %d' %
        #      (labels_x_cpu.int().sum(dim=1)[0],classes_to_do_positive[0][y_pred[0]]))

        '''
        import matplotlib.pyplot as plt
        import plotly.plotly as py
        # Learn about API authentication here: https://plot.ly/python/getting-started
        # Find your api_key here: https://plot.ly/settings/api
        plt.hist([np.arange(len(hist_winner_classes[0,:])),hist_winner_classes[0,:]])
        plt.title("hist_winner_classes")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        fig = plt.gcf()
        plot_url = py.plot_mpl(fig, filename='mpl-basic-histogram')
        '''

        # Final prediction
        y_pred = [classes_batch[y_pred[i]] for i,classes_batch in enumerate(selected_classes)]

        # labels gt
        labels_x_cpu = labels_x_cpu.sum(dim=1).cpu().numpy()

        # Compute metric from embeddings
        for i in range(labels_x_cpu.shape[0]):
            if y_pred[i] == labels_x_cpu[i]:
                correct += 1
            total += 1

        #if (n_iter+1) % 5 == 0:
        #    io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0*correct/total))

    io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0*correct/total))
    io.cprint('*** TEST FINISHED ***\n'.format(correct, total, 100.0 * correct / total))
    enc_nn.train()
    metric_nn.train()

    return 100.0 * correct / total

'''
def test_all_symbols(args, model, test_samples=5000, partition='test'):
    io = io_utils.IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint('\n**** TESTING WITH %s ***' % (partition,))

    loader = generator.Generator(args.dataset_root, args, partition=partition, dataset=args.dataset)

    [enc_nn, metric_nn, softmax_module] = model
    enc_nn.eval()
    metric_nn.eval()
    correct = 0
    total = 0
    iterations = int(test_samples/args.batch_size_test)
    for i in range(iterations):

        # The N_way testing must be the same as the training N_way in training.
        # The solution to classify a new sample that we do not know what can be from all
        # the classes is to select compute an embedding for all the classes in the dataset.
        # and then compute the metrics with the same N_way as in the traininig, always keeping in the
        # n_way batch the class with highest confidence from the previous batch.

        max_classes = len(loader.data)
        data = loader.get_task_batch(batch_size=args.batch_size_test, n_way=max_classes,
                                     num_shots=args.test_N_shots, unlabeled_extra=args.unlabeled_extra,
                                     random_replace_classes = False)

        #data2 = loader.get_task_batch(batch_size=args.batch_size_test, n_way=args.test_N_way,num_shots=args.test_N_shots, unlabeled_extra=args.unlabeled_extra,random_replace_classes=False)
        #[x2, labels_x_cpu2, _, _, xi_s2, labels_yi_cpu2, oracles_yi2, hidden_labels2] = data2

        [x, labels_x_cpu, _, _, xi_s, labels_yi_cpu, oracles_yi, hidden_labels] = data

        if args.cuda:
            xi_s = [batch_xi.cuda() for batch_xi in xi_s]
            labels_yi = [label_yi.cuda() for label_yi in labels_yi_cpu]
            oracles_yi = [oracle_yi.cuda() for oracle_yi in oracles_yi]
            hidden_labels = hidden_labels.cuda()
            x = x.cuda()
        else:
            labels_yi = labels_yi_cpu

        xi_s = [Variable(batch_xi) for batch_xi in xi_s]
        labels_yi = [Variable(label_yi) for label_yi in labels_yi]
        oracles_yi = [Variable(oracle_yi) for oracle_yi in oracles_yi]
        hidden_labels = Variable(hidden_labels)
        x = Variable(x)

        # Set in torch array format
        xi_s = torch.stack(xi_s)
        labels_yi = torch.stack(labels_yi)
        oracles_yi = torch.stack(oracles_yi)
        nclasses, batch_size, nchannels, sizex, sizey = xi_s.shape
        # Flatten the arrays to select different possitions of the array
        xi_s = xi_s.view(nclasses*batch_size, nchannels, sizex, sizey)
        labels_yi = labels_yi.view(nclasses*batch_size,nclasses)
        oracles_yi = oracles_yi.view(nclasses * batch_size, nclasses)

        # Compute embedding from x
        z = enc_nn(x)[-1]

        # Here is the tricky part. He need to feed these inputs with batches of size = args.train_N_way.
        # Also is not possible to keep in cuda memory all enc_nn outputs of the 1476 rotated classes.
        # Do mini_batches of size args.train_N_way.
        isFinished = False
        indices_to_process = np.arange(max_classes)[np.newaxis].repeat(args.batch_size,axis=0)
        while not isFinished:
            # select random classes
            indices_class_lst = [np.random.choice(indices_to_process[i], args.test_N_way, replace=False) for i in
                                      range(args.batch_size)]
            indices_class = np.vstack(indices_class_lst).flatten()
            indices_batch = np.arange(args.test_N_way)[:, np.newaxis].repeat(args.batch_size, axis=1).flatten()
            indices_batch_nway = indices_class+(indices_batch*nclasses)

            # Compute the encoding of the selected clases for this n_way
            xi_tmp = xi_s[indices_batch_nway, ...].view(args.test_N_way, batch_size, nchannels, sizex, sizey)
            zi_s = [enc_nn(batch_xi)[-1] for batch_xi in xi_tmp]
            zi_s = torch.stack(zi_s)

            labels_yi_tmp = labels_yi[indices_batch_nway, ...]
            oracles_yi_tmp = oracles_yi[indices_batch_nway, ...]
            pdb.set_trace()
            labels_yi_tmp = labels_yi[indices_batch_nway, ...].view(args.test_N_way, batch_size, nchannels, sizex, sizey)
            labels_yi_s = [labels_yi[batch_xi][:,indices_to_process] for batch_xi in indices_to_process]
            oracles_yi_s = [oracles_yi[batch_xi] for batch_xi in indices_to_process]
            hidden_labels_s = hidden_labels[:,indices_to_process]
            output, out_logits = metric_nn(inputs=[z, zi_s, labels_yi_s, oracles_yi_s, hidden_labels_s])
            y_pred = softmax_module.forward(output)
            pdb.set_trace()
            y_pred = y_pred.data.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)



            isFinished = True





        # Compute metric from embeddings
        output, out_logits = metric_nn(inputs=[z, zi_s, labels_yi, oracles_yi, hidden_labels])
        output = out_logits
        y_pred = softmax_module.forward(output)
        y_pred = y_pred.data.cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        labels_x_cpu = labels_x_cpu.numpy()
        labels_x_cpu = np.argmax(labels_x_cpu, axis=1)

        for row_i in range(y_pred.shape[0]):
            if y_pred[row_i] == labels_x_cpu[row_i]:
                correct += 1
            total += 1

        if (i+1) % 100 == 0:
            io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0*correct/total))

    io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0*correct/total))
    io.cprint('*** TEST FINISHED ***\n'.format(correct, total, 100.0 * correct / total))
    enc_nn.train()
    metric_nn.train()

    return 100.0 * correct / total
'''