import os
import argparse
import torch
import math
from torch.utils.data import DataLoader
from src.dataset import CUB as Dataset
from src.sampler import Sampler
from src.train_sampler import Train_Sampler
from src.utils import count_acc, Averager, csv_write
from model import Embedding_Module, RelationNetwork

from src.test_dataset import CUB as Test_Dataset
from src.test_sampler import Test_Sampler
from tensorboardX import SummaryWriter

" User input value "
TOTAL = 15000  # total step of training
PRINT_FREQ = 50  # frequency of print loss and accuracy at training step
VAL_FREQ = 100  # frequency of model eval on validation dataset
SAVE_FREQ = 100  # frequency of saving model
TEST_SIZE = 200  # fixed

" fixed value "
VAL_TOTAL = 100

writer = SummaryWriter()

def Test_phase(emb_model, rel_model, args, k):
    emb_model.eval()
    rel_model.eval()
    csv = csv_write(args)

    dataset = Test_Dataset(args.dpath)
    test_sampler = Test_Sampler(dataset._labels, n_way=args.nway, k_shot=args.kshot, query=args.query)
    test_loader = DataLoader(dataset=dataset, batch_sampler=test_sampler, num_workers=4, pin_memory=True)

    print('Test start!')
    for i in range(TEST_SIZE):
        for episode in test_loader:
            data = episode.cuda()

            data_shot, data_query = data[:k], data[k:]

            z_s = emb_model(data_shot)  # Embedded feature of support set. Size = [25,64,4,4]
            z_q = emb_model(data_query)  # Embedded feature of query set. Size = [20,64,4,4]
            z_dim = z_q.size(1)  # 64
            z_s = z_s.view(args.nway, args.kshot, z_dim, 4, 4)
            z_s = torch.sum(z_s, dim=1).squeeze(1)  # Get element-wise sum of embedded feature per class. Size = [5,64,4,4]
            z_s_Ext = z_s.unsqueeze(0).repeat(args.query, 1, 1, 1, 1)  # Copy-paste z_s for relation. Size = [20,5,64,4,4]
            z_q_Ext = z_q.unsqueeze(1).repeat(1, args.nway, 1, 1, 1)  # Copy-paste z_q for relation. Size = [20,5,64,4,4]
            rel_pairs = torch.cat((z_s_Ext, z_q_Ext), dim=2).view(-1, z_dim * 2, 4, 4) # concat features to make relation feature. Size = [100,128,4,4]
            relations = rel_model(rel_pairs).view(-1, args.nway)  # Computed relations
            pred = torch.argmax(relations, dim=1)

            # save your prediction as StudentID_Name.csv file
            csv.add(pred)
            z_proto = None;
            logits = None;
            data = None;
            z_s = None;
            relations = None;

    csv.close()
    print('Test finished, check the csv file!')
    exit()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def train(args):
    # the number of N way, K shot images
    k = args.nway * args.kshot

    # Train data loading
    dataset = Dataset(args.dpath, state='train')
    train_sampler = Train_Sampler(dataset._labels, n_way=args.nway, k_shot=args.kshot, query=args.query)
    data_loader = DataLoader(dataset=dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)

    # Validation data loading
    val_dataset = Dataset(args.dpath, state='val')
    val_sampler = Sampler(val_dataset._labels, n_way=args.nway, k_shot=args.kshot, query=args.query)
    val_data_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)

    """ TODO 1.a """
    " Make your own model for Few-shot Classification in 'model.py' file."

    # model setting
    embedding_module = Embedding_Module()
    relation_module = RelationNetwork(input_size=64, hidden_size=8)

    # weight initialization
    embedding_module.apply(weights_init)
    relation_module.apply(weights_init)
    """ TODO 1.a END """

    # pretrained model load
    if args.load_emb_ckpt is not None:
        emb_state_dict = torch.load(args.load_emb_ckpt)
        embedding_module.load_state_dict(emb_state_dict)


    if args.load_rel_ckpt is not None:
        rel_state_dict = torch.load(args.load_rel_ckpt)
        relation_module.load_state_dict(rel_state_dict)

    embedding_module.cuda()
    embedding_module.train()
    relation_module.cuda()
    relation_module.train()

    if args.test_mode == 1:
         Test_phase(embedding_module, relation_module, args, k)

    """ TODO 1.b (optional) """
    " Set an optimizer or scheduler for Few-shot classification (optional) "

    # Default optimizer setting
    optimizer_embed = torch.optim.Adam(embedding_module.parameters(), lr=0.0008)
    optimizer_relation = torch.optim.Adam(relation_module.parameters(), lr=0.0008)
    """ TODO 1.b (optional) END """

    tl = Averager()  # save average loss
    ta = Averager()  # save average accuracy

    # training start
    print('train start')
    for i in range(TOTAL):
        for episode in data_loader:
            #initialize gradient
            optimizer_embed.zero_grad()
            optimizer_relation.zero_grad()

            data, label = [_.cuda() for _ in episode]  # load an episode

            # split an episode images and labels into shots and query set
            # note! data_shot shape is ( nway * kshot, 3, h, w ) not ( kshot * nway, 3, h, w )
            # Take care when reshape the data shot
            data_shot, data_query = data[:k], data[k:]

            label_shot, label_query = label[:k], label[k:]
            label_shot = sorted(list(set(label_shot.tolist())))

            # convert labels into 0-4 values
            label_query = label_query.tolist()
            labels = []
            for j in range(len(label_query)):
                label = label_shot.index(label_query[j])
                labels.append(label)
            labels = torch.tensor(labels).cuda()

            ''' Relation Network '''
            z_s = embedding_module(data_shot)  # Embedded feature of support set. Size = 25*(64,4,4)
            z_q = embedding_module(data_query)  # Embedded feature of query set. Size = 20*(64,4,4)
            z_dim = z_q.size(1) # 64
            z_s = z_s.view(args.nway, args.kshot, z_dim, 4, 4) # Size [5,5,64,4,4]
            z_s = torch.sum(z_s,dim=1).squeeze(1) # Get element-wise sum of embedded feature per class. Size = [5,64,4,4]
            z_s_Ext = z_s.unsqueeze(0).repeat(args.query,1,1,1,1) # Copy-paste z_s for relation. Size = [20,5,64,4,4]
            z_q_Ext = z_q.unsqueeze(1).repeat(1,args.nway,1,1,1) # Copy-paste z_q for relation. Size = [20,5,64,4,4]
            rel_pairs = torch.cat((z_s_Ext, z_q_Ext), dim=2).view(-1, z_dim * 2, 4, 4)
            relations = relation_module(rel_pairs).view(-1, args.nway) # Computed relations
            mse = torch.nn.MSELoss().cuda()
            one_hot_labels = torch.zeros(args.query, args.nway).cuda().scatter_(1, labels.view(-1,1), 1)
            loss = mse(relations, one_hot_labels)

            """ TODO 2 END """

            acc = count_acc(relations, labels)
            writer.add_scalar("Loss/train", loss, i)
            writer.add_scalar("acc/train", acc, i)
            tl.add(loss.item())
            ta.add(acc)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedding_module.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(relation_module.parameters(), 0.5)
            optimizer_embed.step()
            optimizer_relation.step()

            z_q = None; z_q_Ext = None;  loss = None;
            z_s = None; relations = None; rel_pairs = None;
            mse = None; one_hot_labels = None; labels = None;

        if (i+1) % PRINT_FREQ == 0:
            print('train {}, loss={:.4f} acc={:.4f}'.format(i+1, tl.item(), ta.item()))
            # initialize loss and accuracy mean
            tl = None
            ta = None
            tl = Averager()
            ta = Averager()

        # validation start
        if (i+1) % VAL_FREQ == 0:
            print('validation start')
            embedding_module.eval()
            relation_module.eval()
            with torch.no_grad():
                vl = Averager()  # save average loss
                va = Averager()  # save average accuracy
                for j in range(VAL_TOTAL):
                    for episode in val_data_loader:
                        data, label = [_.cuda() for _ in episode]

                        data_shot, data_query = data[:k], data[k:] # load an episode

                        label_shot, label_query = label[:k], label[k:]
                        label_shot = sorted(list(set(label_shot.tolist())))

                        label_query = label_query.tolist()

                        labels = []
                        for idx in range(len(label_query)):
                            label = label_shot.index(label_query[idx])
                            labels.append(label)
                        labels = torch.tensor(labels).cuda()

                        """ TODO 2 ( Same as above TODO 2 ) """
                        """ Train the model 
                        Input:
                            data_shot : torch.tensor, shot images, [args.nway * args.kshot, 3, h, w]
                                        be careful when using torch.reshape or .view functions
                            data_query : torch.tensor, query images, [args.query, 3, h, w]
                            labels : torch.tensor, labels of query images, [args.query]
                        output:
                            loss : torch scalar tensor which used for updating your model
                            logits : A value to measure accuracy and loss
                        """
                        z_s = embedding_module(data_shot)  # Embedded feature of support set. Size = 25*(64,4,4)
                        z_q = embedding_module(data_query)  # Embedded feature of query set. Size = 20*(64,4,4)
                        z_dim = z_q.size(1)  # 64
                        z_s = z_s.view(args.nway, args.kshot, z_dim, 4, 4)
                        z_s = torch.sum(z_s, dim=1).squeeze(1)  # Get element-wise sum of embedded feature per class. Size = [5,64,4,4]
                        z_s_Ext = z_s.unsqueeze(0).repeat(args.query, 1, 1, 1, 1)  # Copy-paste z_s for relation. Size = [20,5,64,4,4]
                        z_q_Ext = z_q.unsqueeze(1).repeat(1, args.nway, 1, 1,1)  # Copy-paste z_q for relation. Size = [20,5,64,4,4]
                        rel_pairs = torch.cat((z_s_Ext, z_q_Ext), dim=2).view(-1, z_dim * 2, 4, 4)
                        relations = relation_module(rel_pairs).view(-1, args.nway)  # Computed relations
                        mse = torch.nn.MSELoss().cuda()
                        one_hot_labels = torch.zeros(args.query, args.nway).cuda().scatter_(1, labels.view(-1, 1), 1)
                        loss = mse(relations, one_hot_labels)
                        """ TODO 2 END """
                        acc = count_acc(relations, labels)
                        writer.add_scalar("Loss/val",loss, i + j)
                        writer.add_scalar("acc/val", acc, i + j)
                        vl.add(loss.item())
                        va.add(acc)
                        z_q = None; z_q_Ext = None; loss = None; z_s = None;
                        relations = None; rel_pairs = None; mse = None; one_hot_labels = None;


                print('val accuracy mean : %.4f' % va.item())
                print('val loss mean : %.4f' % vl.item())
                # initialize loss and accuracy mean
                vl = None
                va = None
                vl = Averager()
                va = Averager()
            embedding_module.train()
            relation_module.train()

        if (i+1) % SAVE_FREQ == 0:
            embedding_PATH = 'checkpoints/%d_%s.pth' % (i + 1, "embedding")
            relation_PATH = 'checkpoints/%d_%s.pth' % (i + 1, "relation")
            torch.save(embedding_module.state_dict(), embedding_PATH)
            torch.save(relation_module.state_dict(), relation_PATH)
            print('model saved, iteration : %d' % i)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='model', help="name your experiment")
    parser.add_argument('--dpath', '--d', default='./dataset/CUB_200_2011/CUB_200_2011', type=str,
                        help='the path where dataset is located')
    parser.add_argument('--load_emb_ckpt', type=str, help="load embedding checkpoint")
    parser.add_argument('--load_rel_ckpt', type=str, help="load relation checkpoint")
    parser.add_argument('--nway', '--n', default=5, type=int, help='number of class in the support set (5 or 20)')
    parser.add_argument('--kshot', '--k', default=5, type=int,
                        help='number of data in each class in the support set (1 or 5)')
    parser.add_argument('--query', '--q', default=20, type=int, help='number of query data')
    parser.add_argument('--ntest', default=100, type=int, help='number of tests')
    parser.add_argument('--gpus', type=int, nargs='+', default=0)
    parser.add_argument('--test_mode', type=int, default=0, help="if you want to test the model, change the value to 1")

    args = parser.parse_args()

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    torch.cuda.set_device(args.gpus)

    train(args)
    writer.close()
