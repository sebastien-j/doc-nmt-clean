import numpy
import os

from baseline_nmt import train

def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words=params['n_words'][0],
                     n_words_src=params['n_words_src'][0],
                     decay_c=params['decay_c'][0],
                     clip_c=params['clip_c'][0],
                     lrate=params['learning_rate'][0],
                     optimizer=params['optimizer'][0], 
                     patience=1000,
                     maxlen=400,
                     batch_size=60,
                     valid_batch_size=60,
                     validFreq=1000,
                     dispFreq=100,
                     saveFreq=1000,
                     sampleFreq=5000,
                     datasets=['data/train.un16.en-es.es.c0.tok.clean.bpe20k.shuf5.1000000', 
                               'data/train.un16.en-es.en.c0.tok.clean.bpe20k.shuf5.1000000'],
                     valid_datasets=['data/devset.un16.en-es.es.c0.tok.clean.bpe20k', 
                                     'data/devset.un16.en-es.en.c0.tok.clean.bpe20k',
                                     'data/devset.un16.en-es.en.c0.tok.clean'],
                     other_datasets=['data/train.un16.en-es.es.c0.tok.clean.bpe20k.shuf5.1000', 
                                     'data/train.un16.en-es.en.c0.tok.clean.bpe20k.shuf5.1000',
                                     'data/train.un16.en-es.en.c0.tok.clean.shuf5.1000'],
                     dictionaries=['data/train.un16.en-es.es.c0.tok.clean.bpe20k.vocab.pkl', 
                                   'data/train.un16.en-es.en.c0.tok.clean.bpe20k.vocab.pkl'],
                     rng=1234,
                     trng=1234,
                     save_inter=True,
                     encoder='gru',
                     decoder='gru_cond_legacy',
                     valid_output='output/valid_output.baseline',
                     other_output='output/other_output.baseline')
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['models/baseline_model.npz'],
        'dim_word': [620],
        'dim': [1000],
        'n_words_src': [21796],
        'n_words': [21513], 
        'optimizer': ['adadelta'],
        'decay_c': [0.], 
        'clip_c': [1.], 
        'learning_rate': [0.0001],
        'reload': [False]})
