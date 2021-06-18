import os
from timeit import default_timer as timer

from bms import *
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import AdamW
from torch.nn.utils.rnn import pack_padded_sequence

from dataset_384 import *
from swin_transformer_model import *
from util import *
from datetime import datetime

import torch.cuda.amp as amp

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from radam import RAdam
from transformers import get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


is_mixed_precision = True  # [True, False]

if is_mixed_precision:
    class AmpNet(Net):
        @torch.cuda.amp.autocast()
        def forward(self, *args):
            return super(AmpNet, self).forward(*args)
else:
    AmpNet = Net


IDENTIFIER = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, pred, target, length):
        target = target[:, 1:]
        L = [l - 1 for l in length]
        pred = pack_padded_sequence(pred, L, batch_first=True).data
        target = pack_padded_sequence(target, L, batch_first=True).data
        pred = pred.log_softmax(dim=self.dim)

        masked_indices = None
        if self.ignore_index >= 0:
            masked_indices = target.eq(self.ignore_index)

        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        loss = torch.sum(-true_dist * pred, dim=self.dim)

        if masked_indices is not None:
            loss.masked_fill_(masked_indices, 0)

        return loss.sum() / float(loss.size(0) - masked_indices.sum())


def do_valid(net, tokenizer, valid_loader):

    valid_probability = []
    valid_truth = []
    valid_length = []
    valid_num = 0

    net.eval()
    start_timer = timer()
    for t, batch in enumerate(valid_loader):
        batch_size = len(batch['index'])
        image  = batch['image' ].cuda()
        token  = batch['token' ].cuda()
        length = batch['length']

        with torch.no_grad():
            logit = net(image, token, length)
            probability = F.softmax(logit, -1)

        valid_num += batch_size
        valid_probability.append(probability.data.cpu().numpy())
        valid_truth.append(token.data.cpu().numpy())
        valid_length.extend(length)
        print('\r %8d / %d  %s'%(valid_num,
                                 len(valid_loader.sampler),
                                 time_to_str(timer() - start_timer,'sec')
                                 ), end='', flush=True)

    assert(valid_num == len(valid_loader.sampler))  # len(valid_loader.dataset))
    probability = np.concatenate(valid_probability)
    predict = probability.argmax(-1)
    truth = np.concatenate(valid_truth)
    length = valid_length

    # ----
    p = probability[:,:-1].reshape(-1,vocab_size)
    t = truth[:, 1:].reshape(-1)

    non_pad = np.where(t != STOI['<pad>'])[0] #& (t!=STOI['<sos>'])
    p = p[non_pad]
    t = t[non_pad]
    loss = np_loss_cross_entropy(p, t)

    # ----
    lb_score = 0
    if 1:
        score = []
        for i, (p, t) in enumerate(zip(predict, truth)):
            t = truth[i][1:length[i]-1]     # in the buggy version, i have used 1 instead of i
            p = predict[i][1:length[i]-1]
            t = tokenizer.one_predict_to_inchi(t)
            p = tokenizer.one_predict_to_inchi(p)
            s = Levenshtein.distance(p, t)
            score.append(s)
        lb_score = np.mean(score)

    return [loss, lb_score]


def do_calc_lb(net, tokenizer, valid_loader):

    valid_probability = []
    valid_truth = []
    valid_length = []
    valid_num = 0

    text = []

    net.eval()
    start_timer = timer()
    for t, batch in enumerate(valid_loader):
        batch_size = len(batch['index'])
        image  = batch['image' ].cuda()
        token  = batch['token' ].cuda()
        length = batch['length']
        InChI = batch['InChI']

        with torch.no_grad():
            #k = net.forward_argmax_decode(image)
            k = net.module.forward_argmax_decode(image)

            k = k.data.cpu().numpy()
            k = tokenizer.predict_to_inchi(k)
            text.extend(k)

        valid_num += batch_size
        valid_truth.extend(InChI)
        print('\r %8d / %d  %s'%(valid_num,
                                 len(valid_loader.sampler),
                                 time_to_str(timer() - start_timer,'sec')
                                 ), end='', flush=True)

    assert(valid_num == len(valid_loader.sampler))  # len(valid_loader.dataset))

    # ----
    lb_score = 0

    lb_score = compute_lb_score(text, valid_truth)
    lb_score = lb_score.mean()
    return [lb_score]#[loss, lb_score]


def run_train():

    seed_torch()

    fold = 3
    out_dir = '../output/fold%d' % fold
    initial_checkpoint = None
    #initial_checkpoint = out_dir + '/checkpoint/01080000_model.pth'

    debug = 0
    start_lr = 0.0001
    batch_size = 32

    # setup  ----------------------------------------
    for f in ['checkpoint', 'train', 'valid', 'backup']:
        os.makedirs(out_dir + '/' + f, exist_ok=True)

    log = Logger()
    log.open(out_dir + '/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\tout_dir  = %s\n' % out_dir)
    log.write('\n')

    ## dataset ------------------------------------

    df_train, df_valid = make_fold('train-%d' % fold)
    temp_train, df_valid = train_test_split(df_valid, test_size=0.2, random_state=42)
    df_valid = df_valid.reset_index(drop=True)
    df_train = pd.concat([df_train, temp_train])
    df_train = df_train.reset_index(drop=True)
    del temp_train

    tokenizer = load_tokenizer()
    train_dataset = BmsDataset(df_train, tokenizer, augment=null_augment_tr)
    valid_dataset = BmsDataset(df_valid, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        batch_size=batch_size,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=null_collate,
    )

    valid_loader = DataLoader(
        valid_dataset,
        #sampler=SequentialSampler(valid_dataset),
        sampler=FixNumSampler(valid_dataset, 40_000),  # [5_000, 200_000]
        batch_size=128,#256,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=null_collate,
    )

    log.write('train_dataset : \n%s\n' % (train_dataset))
    log.write('valid_dataset : \n%s\n' % (valid_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    if is_mixed_precision:
        scaler = amp.GradScaler()
        net = AmpNet().cuda()
    else:
        net = Net().cuda()

    if initial_checkpoint is not None:
        f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        start_iteration = f['iteration']
        start_epoch = f['epoch']
        state_dict = f['state_dict']
        net.load_state_dict(state_dict, strict=True)  # True
    else:
        start_iteration = 0
        start_epoch = 0

    if torch.cuda.device_count() > 1:
        log.write("Let's use %d GPUs! \n" % (torch.cuda.device_count()))
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        net = nn.DataParallel(net)

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    log.write('\n')

    # -----------------------------------------------
    if 0:  ##freeze
        for p in net.encoder.parameters(): p.requires_grad = False

    optimizer = RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr)

    num_iteration = 80000 * 1000
    iter_log = 8000
    iter_valid = 8000
    iter_calc_lb = 40000
    iter_save = list(range(0, num_iteration, 8000))  # 1*1000

    num_warmup_steps=8000
    num_train_steps = 800000
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)

    log.write('optimizer\n  %s\n' % (optimizer))
    log.write('\n')

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('   is_mixed_precision = %s \n' % str(is_mixed_precision))
    log.write('   batch_size = %d\n' % (batch_size))
    log.write('                      |----- VALID ---|---- TRAIN/BATCH --------------\n')
    log.write('     rate     iter   epoch    |    loss    lb(TF)     |    lb(AR)    |    loss0    loss1    | time          \n')
    log.write('----------------------------------------------------------------------\n')
             # 0.00000   0.00* 0.00  | 0.000  0.000  | 0.000  0.000  |  0 hr 00 min

    def message(mode='print'):
        if mode == ('print'):
            asterisk = ' '
            loss = batch_loss
        if mode == ('log'):
            asterisk = '*' if iteration in iter_save else ' '
            loss = train_loss

        text = \
            '%0.5f  %5.4f%s %4.2f  | ' % (rate, iteration / 10000, asterisk, epoch,) + \
            '%4.4f  %5.3f  | ' % (*valid_loss,) + \
            '%5.3f  | ' % (*valid_lb,) + \
            '%4.4f  %4.3f  %4.3f  | ' % (*loss,) + \
            '%s' % (time_to_str(timer() - start_timer, 'min'))

        return text

    # ----
    valid_loss = np.zeros(2, np.float32)
    valid_lb = np.zeros(1, np.float32)
    train_loss = np.zeros(3, np.float32)
    batch_loss = np.zeros_like(train_loss)
    sum_train_loss = np.zeros_like(train_loss)
    sum_train = 0
    loss0 = torch.FloatTensor([0]).cuda().sum()
    loss1 = torch.FloatTensor([0]).cuda().sum()
    loss2 = torch.FloatTensor([0]).cuda().sum()

    start_timer = timer()
    iteration = start_iteration
    epoch = start_epoch
    rate = 0
    while iteration < num_iteration:

        for t, batch in enumerate(train_loader):

            if iteration in iter_save:
                #if iteration != start_iteration:
                    torch.save({
                        'state_dict': net.module.state_dict(),
                        #'state_dict': net.state_dict(),
                        'iteration': iteration,
                        'epoch': epoch,
                    }, out_dir + '/checkpoint/%08d_model.pth' % (iteration))
                    pass

            if (iteration % iter_valid == 0):
                if iteration != start_iteration:
                    valid_loss = do_valid(net, tokenizer, valid_loader)
                    pass

            if (iteration % iter_calc_lb == 0):
                if iteration != start_iteration:
                    valid_lb = do_calc_lb(net, tokenizer, valid_loader)
                    pass

            if (iteration % iter_log == 0):
                print('\r', end='', flush=True)
                log.write(message(mode='log') + '\n')

            # learning rate schduler ------------
            rate = get_learning_rate(optimizer)

            # one iteration update  -------------
            batch_size = len(batch['index'])
            image  = batch['image' ].cuda()
            token  = batch['token' ].cuda()
            length = batch['length']

            # ----
            net.train()
            optimizer.zero_grad()

            if is_mixed_precision:
                with amp.autocast():
                    #assert(False)
                    logit = net(image, token, length)
                    loss0 = seq_cross_entropy_smoothing_loss(logit, token, length)

                scaler.scale(loss0).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            else:
                assert False
                # print('fp32')
                # image_embed = encoder(image)
                logit, weight = decoder(image_embed, token, length)

                (loss0).backward()
                optimizer.step()

            # print statistics  --------
            epoch += 1 / len(train_loader)
            iteration += 1

            batch_loss = np.array([loss0.item(), loss1.item(), loss2.item()])
            sum_train_loss += batch_loss
            sum_train += 1
            if iteration % 100 == 0:
                train_loss = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss[...] = 0
                sum_train = 0

            print('\r', end='', flush=True)
            print(message(mode='print'), end='', flush=True)

            # debug--------------------------
            if debug:
                pass

    log.write('\n')


seq_cross_entropy_smoothing_loss = LabelSmoothingLoss(193, smoothing=0.1, dim=-1, ignore_index=192)


if __name__ == '__main__':
    run_train()
