To train a baseline model: `python train_baseline_nmt.py`
To train a doc-level model: `python train_doc_nmt.py`

To translate a source file with a baseline modeL:
    `python baseline_translate.py -k 5 -p 5 $baseline_model $src_dict $trg_dict $src $baseline_output`
With a doc-level model:
    `python doc_translate.py -k 5 -p 5 $doc_model $src_dict $trg_dict $src $ctx $doc_output 100`

To obtain log probs with a baseline model:
    `python baseline_translate.py $baseline_model $src_dict $trg_dict $src $baseline_output $baseline_logprobs`
With a doc-level model:
    `python doc_translate.py $doc_model $src_dict $trg_dict $src $doc_output $ctx $doc_logprobs`

If you use byte-pair encoding, you may merge the tokens with:
    `sed "s/@@ //g" < $baseline_output > ${baseline_output}.words`
    `sed "s/@@ //g" < $doc_output > ${doc_output}.words`

To combine the outputs:
    `python combine_output.py ${baseline_output}.words ${doc_output}.words $baseline_logprobs $doc_logprobs $combined_output`
