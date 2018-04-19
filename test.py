import numpy as np
from utils import io_utils
from data import generator
from torch.autograd import Variable
import pdb

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

        # Compute embedding from x
        z = enc_nn(x)[-1]

        # Here is the tricky part. He need to feed these inputs with batches of size = args.train_N_way.
        # Also is not possible to keep in cuda memory all enc_nn outputs of the 1476 rotated classes.
        # Do mini_batches of size args.train_N_way.
        isFinished = False

        indices_to_process = np.arange(max_classes)
        while not isFinished:
            # select random classes
            indices_to_process = np.random.choice(indices_to_process,args.test_N_way,replace=False)
            # selected classes
            zi_s = [enc_nn(xi_s[batch_xi])[-1] for batch_xi in indices_to_process]
            labels_yi_s = [labels_yi[batch_xi] for batch_xi in indices_to_process]
            oracles_yi_s = [oracles_yi[batch_xi] for batch_xi in indices_to_process]
            hidden_labels_s = hidden_labels[:,indices_to_process]
            pdb.set_trace()
            output, out_logits = metric_nn(inputs=[z, zi_s, labels_yi_s, oracles_yi_s, hidden_labels_s])
            y_pred = softmax_module.forward(output)
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
