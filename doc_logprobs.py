import argparse

import numpy
import theano
import cPickle as pkl

from doc_nmt import (load_params, init_params, init_tparams,
    build_model, pred_probs, prepare_data)

from data_iterator import TextIterator

profile = False

def main(model, dictionary, dictionary_target, source, target, context, outfile, wordbyword):

    # load model model_options
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)

    valid_noshuf = TextIterator(source, target, context,
                         dictionary, dictionary_target,
                         n_words_source=options['n_words_src'], n_words_target=options['n_words'],
                         batch_size=options['valid_batch_size'], maxlen=2000, shuffle=False,
                         tc=options['kwargs'].get('tc', False))

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, xc, xc_mask, \
        opt_ret, \
        cost, cost_, xc_mask_2, xc_mask_3 = \
        build_model(tparams, options)
    inps = [x, x_mask, y, y_mask, xc, xc_mask, xc_mask_2, xc_mask_3]

    f_log_probs = theano.function(inps, cost, profile=profile)

    valid_errs = pred_probs(f_log_probs, prepare_data, options, valid_noshuf, verbose=True)
    numpy.save(outfile, valid_errs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('target', type=str)
    parser.add_argument('context', type=str)
    parser.add_argument('outfile', type=str)

    args = parser.parse_args()

    main(args.model, args.dictionary, args.dictionary_target,
         args.source, args.target, args.context, args.outfile)

