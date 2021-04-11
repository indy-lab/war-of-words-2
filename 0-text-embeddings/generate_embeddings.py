import json
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
import re
import fasttext
import argparse

def _filter_dossiers(dataset, thr):
    # Count occurence of each dossiers.
    dossiers = list()
    for data in dataset:
        for datum in data:
            dossiers.append(datum['dossier_ref'])
    counter = Counter(dossiers)
    # Define list of dossiers to keep.
    keep = set([d for d, c in counter.items() if c > thr])
    k, d = len(keep), len(set(dossiers))
    print(f'Removed {d-k} ({(d-k)/d*100:.2f}%) dossiers.')
    return keep
def _filter_meps(dataset, thr):
    # Count occurence of each dossiers.
    meps = list()
    for data in dataset:
        for datum in data:
            for at in datum['authors']:
                meps.append(at['id'])
    counter = Counter(meps)
    # Define list of dossiers to keep.
    keep = set([d for d, c in counter.items() if c > thr])
    k, m = len(keep), len(set(meps))
    print(f'Removed {m-k} ({(m-k)/m*100:.2f}%) MEPs.')
    return keep
def filter_dataset(dataset, thr=10):
    """Remove dossiers with less than `thr` edits."""
    keep_doss = _filter_dossiers(dataset, thr)
    keep_mep = _filter_meps(dataset, thr)
    filtered_dataset = list()
    for data in dataset:
        kd, km = True, True
        for datum in data:
            if datum['dossier_ref'] not in keep_doss:
                kd = False
            if not all(at['id'] in keep_mep for at in datum['authors']):
                km = False
        if kd and km:
            filtered_dataset.append(data)
    d, f = len(dataset), len(filtered_dataset)
    print(f'Removed {d-f} ({(d-f)/d*100:.2f}%) conflicts.')
    print('Number of data points:', len(filtered_dataset))
    return filtered_dataset
def _shuffle(dataset, seed):
    np.random.seed(seed)
    perm = np.random.permutation(range(len(dataset)))
    return [dataset[p] for p in perm], perm

def unroll(dataset):
    unrolled = list()
    for conflict in dataset:
        for edit in conflict:
            unrolled.append(edit)
    return unrolled


def load_data(legislature,task,data_dir,dossier2title_dir):
    # Loading canonical dataset
    if task=='new_edit':
        data = []
        with open(data_dir+'/war-of-words-2-ep' + legislature + '.txt','r') as json_file:
            for line in json_file:
                data.append(json.loads(line))
    elif task=='new_dossier':
        data = []
        with open(data_dir+'/war-of-words-2-ep' + legislature + '-chronological.txt','r') as json_file:
            for line in json_file:
                data.append(json.loads(line))
    else:
        print('Invalid task')
        return
    
    # Loading mapping of dossier references to title into a dictionary

    with open(dossier2title_dir + '/dossier-titles.json','r') as json_file:
        ref2title = json.load(json_file)
            
            
    # Adding dossier title to the edits and keeping track of missing dossiers

    missing_refs = set()
    for conflict in data:
        for edit in conflict:
            dossier_ref = edit['dossier_ref']
            if dossier_ref in ref2title:
                edit['dossier_title'] = ref2title[dossier_ref]
            else:
                edit['dossier_title'] = ""
                missing_refs = missing_refs.union({dossier_ref})

    if len(missing_refs) > 0:
        print('Warning !', len(missing_refs),'references do not have an associated title!')
    
    # Unrolling dataset
    
    data_unrolled = unroll(data)
            
    return data, data_unrolled

def split_shuffle_data(legislature,task,data,indices_dir=None,seed=0):

    if task=='new_dossier':
        dat_train = unroll(data[:(112339+13218)])
        dat_test = unroll(data[(112339+13218):])
    elif task=='new_edit':
        if indices_dir is None:
            print('Indices directory not given')
            return
        train_indices = np.loadtxt(indices_dir + '/ep' + legislature +'-train-indices.txt',dtype=int)
        test_indices = np.loadtxt(indices_dir + '/ep' + legislature +'-test-indices.txt',dtype=int)
        data_filtered = filter_dataset(data)
        dat_train = unroll(_shuffle([data_filtered[idx] for idx in train_indices],seed=seed)[0])
        dat_test = unroll(_shuffle([data_filtered[idx] for idx in test_indices],seed=seed)[0])
    else:
        print('Invalid task')
        return
    
    return dat_train,dat_test

def write_to_txt(data_,dat_type='train'):
    fname_edit = dat_type + '_data_edit.txt'
    fname_title = dat_type + '_data_title.txt'
    with open(fname_edit,'w',encoding='utf8') as f:
        print('Writing edit data')
        for datum in data_:
            i1 = datum['edit_indices']['i1']
            i2 = datum['edit_indices']['i2']
            j1 = datum['edit_indices']['j1']
            j2 = datum['edit_indices']['j2']

            text_del = datum['text_original'][i1:i2]
            text_ins = datum['text_amended'][j1:j2]
            text_context_l = datum['text_original'][:i1] 
            text_context_r = datum['text_original'][i2:]

            f.write('__label__'+str(datum['accepted'])+ ' <con>'+ ' <con>'.join(text_context_l) + ' <del>' + ' <del>'.join(text_del) + ' <con>'+ ' <con>'.join(text_context_r) +' <ins>'+ ' <ins>'.join(text_ins)  +  '\n')       
    with open(fname_title,'w',encoding='utf8') as f:
        print('Writing title data')
        for datum in data_:
            f.write('__label__'+str(datum['accepted'])+ ' ' + ' '.join([re.sub('\d','D',w.lower()) for w in word_tokenize(datum['dossier_title'])])  +'\n')         

def get_hyperparameters(legislature,task):
    if task=='new_dossier':
        epochs_edit = 10
        lr_edit = 0.005
        epochs_title = 12
        lr_title = 0.0001
    elif task=='new_edit':
        if legislature == '8':
            epochs_edit = 3
            lr_edit = 0.1
            epochs_title = 10
            lr_title = 0.1
        elif legislature == '7':
            epochs_edit = 3
            lr_edit = 0.05
            epochs_title = 10
            lr_title = 0.1
        else:
            print('Invalid legislature')
            return
    else:
        print('Invalid task')
        return     
    return epochs_edit,lr_edit, epochs_title, lr_title


def train_save_models(legislature, task, epochs_edit,lr_edit,epochs_title,lr_title,output_dir,model_type='train'):

    model_edit = fasttext.train_supervised(input=model_type+'_data_edit.txt',epoch=epochs_edit,dim=10,wordNgrams=2,lr=lr_edit)

    model_title = fasttext.train_supervised(input=model_type+'_data_title.txt',epoch=epochs_title,dim=10,wordNgrams=2,lr=lr_title)

    model_edit.save_model(output_dir + '/ep' + legislature + '-'+task+'-' + model_type + '-edit.bin')

    model_title.save_model(output_dir + '/ep'+ legislature + '-'+task+'-' + model_type + '-title.bin')
    
    return model_edit, model_title


def gen_save_embeddings(legislature, task, model_edit, model_title, data_unrolled,output_dir,embed_type='train'):
    '''
    Generate and save the representations of the edits and titles
    '''
    feats_edit = np.zeros((len(data_unrolled),10))
    feats_title = np.zeros((len(data_unrolled),10))
    for i,datum in enumerate(data_unrolled):
        i1 = datum['edit_indices']['i1']
        i2 = datum['edit_indices']['i2']
        j1 = datum['edit_indices']['j1']
        j2 = datum['edit_indices']['j2']
        
        text_del = datum['text_original'][i1:i2]
        text_ins = datum['text_amended'][j1:j2]
        text_context_l = datum['text_original'][:i1] 
        text_context_r = datum['text_original'][i2:]

        text_datum = '<con>'+ ' <con>'.join(text_context_l) + ' <del>' + ' <del>'.join(text_del) + ' <con>'+ ' <con>'.join(text_context_r) +' <ins>'+ ' <ins>'.join(text_ins)
        feats_edit[i,:] = model_edit.get_sentence_vector(text_datum)

        text_datum_title = ' '.join([re.sub('\d','D',w.lower()) for w in word_tokenize(datum['dossier_title'])])
        feats_title[i,:] = model_title.get_sentence_vector(text_datum_title)
    
    suffix = ''

    if task=='new_dossier':
        suffix='-chronological'
    elif embed_type=='full':
        suffix='-full'

    np.savetxt(output_dir+'/ep'+ legislature + '-edit_embedding' + suffix + '.txt', feats_edit)
    np.savetxt(output_dir+'/ep'+ legislature + '-title_embedding' + suffix + '.txt', feats_title)


def main(args):
    legislature = args.leg
    if args.chronological:
        if legislature=='8':
            task='new_dossier'
        else:
            print('Task is not applicable for 7th legislature!')
            return
    else:
        task='new_edit'

    print('Task: ',task, ' Legislature: ', legislature)

    # Load data
    print('Loading data...')     
    data, data_unrolled = load_data(legislature, task, args.data_dir, args.dossier2title_dir)

    # Split into train and test sets
    dat_train, dat_test = split_shuffle_data(legislature, task, data, args.indices_dir)

    # Write dataset as text files for training fasttext models
    print('Writing training data to text...')
    write_to_txt(dat_train,dat_type='train')
    print('Writing test data to text...')
    write_to_txt(dat_test,dat_type='test')
    print('Writing full data to text...')
    write_to_txt(dat_train+dat_test,dat_type='full')

    # Get hyperparams for training
    epochs_edit, lr_edit, epochs_title, lr_title = get_hyperparameters(legislature=legislature,task=task)

    # Train and save models - on train set and on full set (for interpretation)
    print('Training and saving models...')
    model_edit, model_title = train_save_models(legislature, task, epochs_edit, lr_edit, epochs_title, lr_title, output_dir=args.text_embedding)
    if task=='new_edit' and legislature=='8':
        fullmodel_edit, fullmodel_title = train_save_models(legislature, task, epochs_edit, lr_edit, epochs_title, lr_title, output_dir=args.text_embedding, model_type='full')

    # Generate 
    print('Generating and saving embeddings...')
    gen_save_embeddings(legislature, task, model_edit=model_edit,model_title=model_title,data_unrolled=data_unrolled, output_dir=args.text_embedding)
    if task=='new_edit' and legislature=='8':
        gen_save_embeddings(legislature, task, model_edit=model_edit,model_title=model_title,data_unrolled=data_unrolled, output_dir=args.text_embedding, embed_type='full')

    print('All done !')

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--leg', help='Legislature term(7 or 8)'
    )
    parser.add_argument('--data_dir', help='Path to directory of data')
    parser.add_argument('--indices_dir', help='Path to directory of train and test indices')
    parser.add_argument('--dossier2title_dir', help='Path to directory containing mapping from dossier to title')
    parser.add_argument('--text_embeddings_dir', help='Path to directory to store the final text embeddings and trained models')
    parser.add_argument(
        '--chronological',
        action='store_true',
        help='Task is chronological',
    )
    main(parser.parse_args())

