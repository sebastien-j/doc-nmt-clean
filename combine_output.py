import argparse
import numpy

def main(output1, output2, logprobs1, logprobs2, combined):
    lp1 = numpy.load(logprobs1)
    lp2 = numpy.load(logprobs2)

    with open(output1) as f:
        with open(output2) as g:
            with open(combined, 'w') as h:
                for ii, fline in enumerate(f):
                    gline = g.readline()
                    fscore = lp1[ii]/(len(fline.split()) + 1)
                    gscore = lp2[ii]/(len(gline.split()) + 1)
                    if fscore < gscore:
                        h.write(fline)
                    else:
                        h.write(gline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output1')
    parser.add_argument('--output2')
    parser.add_argument('--logprobs1')
    parser.add_argument('--logprobs2')
    parser.add_argument('--combined')

    args = parser.parse_args()

    main(args.output1, args.output2, args.logprobs1, args.logprobs2, args.combined)
