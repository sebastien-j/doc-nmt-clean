'''
Translates a source file using a translation model.
'''
import argparse

import numpy
import theano
import cPickle as pkl

from doc_nmt import (build_sampler, gen_sample, load_params,
                 init_params, init_tparams)

from multiprocessing import Process, Queue


def translate_model(queue, rqueue, pid, model, options, k, normalize, maxlen):

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(options['trng'])
    use_noise = theano.shared(numpy.float32(0.))

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    # word index
    f_init, f_next = build_sampler(tparams, options, trng, use_noise)

    def _translate(seq, seq_context, xc_mask, xc_mask_2, xc_mask_3):
        # sample given an input sequence and obtain scores
        sample, score = gen_sample(tparams, f_init, f_next,
                                   numpy.array(seq).reshape([len(seq), 1]),
                                   xc, xc_mask, xc_mask_2, xc_mask_3,
                                   options, trng=trng, k=k, maxlen=maxlen,
                                   stochastic=False, argmax=False)

        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths
        sidx = numpy.argmin(score)
        return sample[sidx]

    while True:
        req = queue.get()
        if req is None:
            break

        idx, x, xc, xc_mask, xc_mask_2, xc_mask_3 = req[0], req[1], req[2], req[3], req[4], req[5]
        print pid, '-', idx
        seq = _translate(x, xc, xc_mask, xc_mask_2, xc_mask_3)

        rqueue.put((idx, seq))

    return


def main(model, dictionary, dictionary_target, source_file, source_context_file, saveto, k=5,
         normalize=False, n_process=5, maxlen=100):

    # load model model_options
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)

    print options['kwargs']
    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    # create input and output queues for processes
    queue = Queue()
    rqueue = Queue()
    processes = [None] * n_process
    for midx in xrange(n_process):
        processes[midx] = Process(
            target=translate_model,
            args=(queue, rqueue, midx, model, options, k, normalize, maxlen))
        processes[midx].start()

    # utility function
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                ww.append(word_idict_trg[w])
            capsw.append(' '.join(ww))
        return capsw

    def _send_jobs(fname, gname):
        with open(fname, 'r') as f:
            with open(gname, 'r') as g:
                for idx, line in enumerate(f):
                    words = line.strip().split()
                    x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                    x = map(lambda ii: ii if ii < options['n_words_src'] else 1, x)
                    x += [0]

                    gline = g.readline()
                    context_words = gline.strip().split(' ||| ') # with |||
                    
                    num_sent = len(context_words)
                    maxlen_ = max(len(sent.strip().split()) for sent in context_words) + 1
                    xc = numpy.zeros((num_sent, 1, maxlen_), dtype=numpy.int64)
                    xc_mask = numpy.zeros((num_sent, 1, maxlen_), dtype=numpy.float32)
                    xc_mask_2 = numpy.zeros((num_sent, 1), dtype=numpy.float32)
                    xc_mask_3 = numpy.zeros((num_sent, 1), dtype=numpy.float32)

                    for ii, sent in enumerate(context_words):
                        sent = sent.strip().split()
                        tmp = map(lambda w: word_dict[w] if w in word_dict else 1, sent)
                        tmp = map(lambda ii: ii if ii < options['n_words_src'] else 1, tmp)
                        tmp += [0]
                        xc[ii,0,:len(tmp)] = tmp
                        xc_mask[ii,0,:len(tmp)] = 1.0
                        xc_mask_2[ii,0] = 1.0
                        xc_mask_3[ii,0] = float(len(tmp))
                    
                    queue.put((idx, x, xc, xc_mask, xc_mask_2, xc_mask_3))
        return idx+1

    def _finish_processes():
        for midx in xrange(n_process):
            queue.put(None)

    def _retrieve_jobs(n_samples):
        trans = [None] * n_samples
        for idx in xrange(n_samples):
            resp = rqueue.get()
            trans[resp[0]] = resp[1]
            if numpy.mod(idx, 10) == 0:
                print 'Sample ', (idx+1), '/', n_samples, ' Done'
        return trans

    print 'Translating ', source_file, 'with context from', source_context_file, '...'
    n_samples = _send_jobs(source_file, source_context_file)
    trans = _seqs2words(_retrieve_jobs(n_samples))
    _finish_processes()
    with open(saveto, 'w') as f:
        print >>f, '\n'.join(trans)
    print 'Done'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('-p', type=int, default=5)
    parser.add_argument('-n', action="store_true", default=False)
    parser.add_argument('model', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('source_context', type=str)
    parser.add_argument('saveto', type=str)
    parser.add_argument('maxlen', type=int, default=100)

    args = parser.parse_args()
    print args.ctx_type

    main(args.model, args.dictionary, args.dictionary_target, args.source, args.source_context,
         args.saveto, k=args.k, normalize=args.n, n_process=args.p, maxlen=args.maxlen)
