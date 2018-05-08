import numpy as np
from sklearn.metrics import roc_curve

## making dictionary to find blacklist pair between train and test dataset
bl_match = np.loadtxt('data/bl_matching_dev.csv',dtype='string')
dev2train={}
dev2id={}
train2dev={}
train2id={}
for iter, line in enumerate(bl_match):
    line_s = line.split(',')
    dev2train[line_s[1].split('_')[-1]]= line_s[2].split('_')[-1]
    dev2id[line_s[1].split('_')[-1]]= line_s[0].split('_')[-1]
    train2dev[line_s[2].split('_')[-1]]= line_s[1].split('_')[-1]
    train2id[line_s[2].split('_')[-1]]= line_s[0].split('_')[-1]
    
def load_ivector(filename):
    spk_id = np.loadtxt(filename,dtype='string',delimiter=',',skiprows=1,usecols=[0])
    ivector = np.loadtxt(filename,dtype='float32',delimiter=',',skiprows=1,usecols=range(1,601))

    for iter in range(len(spk_id)):
        spk_id[iter] = spk_id[iter].split('_')[0]

    return spk_id, ivector

def length_norm(mat):
# length normalization (l2 norm)
# input: mat = [utterances X vector dimension] ex) (float) 8631 X 600

    norm_mat = []
    for line in mat:
        temp = line/np.math.sqrt(sum(np.power(line,2)))
        norm_mat.append(temp)
    norm_mat = np.array(norm_mat)
    return norm_mat

def make_spkvec(mat, spk_label):
# calculating speaker mean vector
# input: mat = [utterances X vector dimension] ex) (float) 8631 X 600
#        spk_label = string vector ex) ['abce','cdgd']

#     for iter in range(len(spk_label)):
#         spk_label[iter] = spk_label[iter].split('_')[0]

    spk_label, spk_index  = np.unique(spk_label,return_inverse=True)
    spk_mean=[]
    mat = np.array(mat)

    # calculating speaker mean i-vector
    for i, spk in enumerate(spk_label):
        spk_mean.append(np.mean(mat[np.nonzero(spk_index==i)],axis=0))
    spk_mean = length_norm(spk_mean)
    return spk_mean, spk_label

def calculate_EER(trials, scores):
# calculating EER of Top-S detector
# input: trials = boolean(or int) vector, 1: postive(blacklist) 0: negative(background)
#        scores = float vector

    # Calculating EER
    fpr,tpr,threshold = roc_curve(trials,scores,pos_label=1)
    fnr = 1-tpr
    EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    
    # print EER_threshold
    EER = fpr[np.argmin(np.absolute((fnr-fpr)))]
    print "Top S detector EER is %0.2f%%"% (EER*100)
    return EER

def get_trials_label_with_confusion(identified_label, groundtruth_label,dict4spk ):
# determine if the test utterance would make confusion error
# input: identified_label = string vector, identified result of test utterance among multi-target from the detection system 
#        groundtruth_label = string vector, ground truth speaker labels of test utterances
#        dict4spk = dictionary, convert label to target set, ex) train2dev convert train id to dev id

    trials = np.zeros(len(identified_label))
    for iter in range(0,len(groundtruth_label)):
        enroll = identified_label[iter].split('_')[0]
        test = groundtruth_label[iter].split('_')[0]
        try: 
            dict4spk[test]
            if enroll == dict4spk[test]:
                trials[iter]=1 # for Target trial (blacklist speaker)
            else:
                trials[iter]=-1 # for Target trial (backlist speaker), but fail on blacklist classifier
        except KeyError:
            trials[iter]=0 # for non-target (non-blacklist speaker)
    return trials


def calculate_EER_with_confusion(scores,trials):
# calculating EER of Top-1 detector
# input: trials = boolean(or int) vector, 1: postive(blacklist) 0: negative(background) -1: confusion(blacklist)
#        scores = float vector

    # exclude confusion error (trials==-1)
    scores_wo_confusion = scores[np.nonzero(trials!=-1)[0]]
    trials_wo_confusion = trials[np.nonzero(trials!=-1)[0]]

    # dev_trials contain labels of target. (target=1, non-target=0)
    fpr,tpr,threshold = roc_curve(trials_wo_confusion,scores_wo_confusion,pos_label=1, drop_intermediate=False)
    fnr = 1-tpr
    EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    
    # EER withouth confusion error
    EER = fpr[np.argmin(np.absolute((fnr-fpr)))]
    
    # Add confusion error to false negative rate(Miss rate)
    total_negative = len(np.nonzero(np.array(trials_wo_confusion)==0)[0])
    total_positive = len(np.nonzero(np.array(trials_wo_confusion)==1)[0])
    fp= fpr*np.float(total_negative)  
    fn= fnr*np.float(total_positive) 
    fn += len(np.nonzero(trials==-1)[0])
    total_positive += len(np.nonzero(trials==-1)[0])
    fpr= fp/total_negative
    fnr= fn/total_positive

    # EER with confusion Error
    EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    EER = fpr[np.argmin(np.absolute((fnr-fpr)))]
    
    print "Top 1 detector EER is %0.2f%% (Total confusion error is %d)"% ((EER*100), len(np.nonzero(trials==-1)[0]))
    return EER

# Loading i-vector
trn_bl_id, trn_bl_ivector = load_ivector('data/trn_blacklist.csv')
trn_bg_id, trn_bg_ivector = load_ivector('data/trn_background.csv')
dev_bl_id, dev_bl_ivector = load_ivector('data/dev_blacklist.csv')
dev_bg_id, dev_bg_ivector = load_ivector('data/dev_background.csv')

# Calculating speaker mean vector
spk_mean, spk_mean_label = make_spkvec(trn_bl_ivector,trn_bl_id)

#length normalization
trn_bl_ivector = length_norm(trn_bl_ivector)
trn_bg_ivector = length_norm(trn_bg_ivector)
dev_bl_ivector = length_norm(dev_bl_ivector)
dev_bg_ivector = length_norm(dev_bg_ivector)


print '\nDev set score using train set :'
# making trials of Dev set
dev_ivector = np.append(dev_bl_ivector, dev_bg_ivector,axis=0)
dev_trials = np.append( np.ones([len(dev_bl_id), 1]), np.zeros([len(dev_bg_id), 1]))

# Cosine distance scoring
scores = spk_mean.dot(dev_ivector.transpose())
dev_scores = np.max(scores,axis=0)

# Top-S detector EER
dev_EER = calculate_EER(dev_trials, dev_scores)

#divide trial label into target and non-target, plus confusion error(blacklist, fail at blacklist detector)
dev_identified_label = spk_mean_label[np.argmax(scores,axis=0)]
dev_trials_label = np.append( dev_bl_id,dev_bg_id)

# Top-1 detector EER
dev_trials_confusion = get_trials_label_with_confusion(dev_identified_label, dev_trials_label, dev2train )
dev_EER_confusion = calculate_EER_with_confusion(dev_scores,dev_trials_confusion)

# Generating submission file on Dev set for example
filename = 'teamname_fixed_primary.csv'
# filename = 'teamname_fixed_contrastive1.csv'
with open(filename, "w") as text_file:
    for iter,score in enumerate(dev_scores):
        id_in_trainset = dev_identified_label[iter].split('_')[0]
        input_file = dev_trials_label[iter]
        text_file.write('%s,%s,%s\n' % (input_file,score,train2id[id_in_trainset]))

    