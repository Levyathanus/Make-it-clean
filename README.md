# References
**Project idea:** Make it clean (P9) [see the original draft below].

**Author:** Michele Zenoni

**Course:** Information Retrieval (A.A. 2022/2023) - Master Degree in Computer Science

**Teachers:** Prof. Alfio Ferrara

Dott. Sergio Picascia, Dott. Davide Riva, Dott.ssa 
Elisabetta Rocchetti, Dott.ssa Darya Shlyk

> **_DISCLAIMER:_** no generative LLMs were used to write the project paper, nor
the project presentation.

# [Make it clean (P9)](https://contents.islab.di.unimi.it/teaching/courseprojects/inforet-projects-2022-23.html#make-it-clean-p9)
Depending on the quality of the original document, Optical Character Recognition (OCR) can produce a range of errors – from erroneous letters to additional and spurious blank spaces. This issue risks to compromise the effectiveness of the analysis tasks in support to the study of texts. Furthermore, the presence of multiple errors of different type in a certain text segment can introduce so much noise that the overall digitization process becomes useless.

The goal of this project is to explore machine learning techniques, in particular sequence to sequence learning, to develop a text correction tool. 

Typically, there are two main errors:

1. wrong characters in words: ```this senlence contains and effor```

2. wrong segmentation: ```th is sentencecont ains an error```

They are usually combined, but the project can be focused on only one of the two. The first is more easy to correct using regular expressions and rule-based techniques. The second is more tricky.

Here we give some possible options to start with, but any original idea is welcome:

Given a sentence $s$, containing OCR errors and wrong word segmentation, we can:

- **Detect candidate wrong words:** using a (eventually pre-trained) language model, we can associate each token of with a probability of being a correct word in the model vocabulary. Then we can start from the wrong words and try to correct them, maybe by recursively trying to concatenate wrong words for the wrong segmentation problem.  

- **Seq2seq:** developing and training a seq2seq model base on single chars or n-grams of chars it is possible to build a spelling correction system that may help in cleaning the text.

- **Transformers tokenizers:** implementing an iterative learning model for tokenization (inspired to the ones used in transformers) we could produce a special tokenizer that can correct the text.

## Dataset
Any dataset fictional work can be used for the project, provided of course that the text is available. In this case, you need to buld a model that introduces fake errors in the text in order to have the original text as a ground truth and the perturbated text as the text to be cleaned. It is suggested to search also for datasets that provide annotated fictional works in order to help in evaluating the model.

Another option is to get a collection of (possibly old) pdf and use OCR (like Tesseract) to create the dataset.

A dataset made of real Parliamentary acts in Italian is available. Contact Prof. Ferrara for this option.

## References
Nastase, V., & Hitschler, J. (2018, May). Correction of OCR word segmentation errors in articles from the ACL collection through neural machine translation methods. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018).

Todorov, K., & Colavizza, G. (2020). Transfer Learning for Historical Corpora: An Assessment on Post-OCR Correction and Named Entity Recognition. Proceedings http://ceur-ws.org ISSN, 1613, 0073.

Nguyen, T. T. H., Jatowt, A., Nguyen, N. V., Coustaty, M., & Doucet, A. (2020, August). Neural Machine Translation with BERT for Post-OCR Error Detection and Correction. In Proceedings of the ACM/IEEE Joint Conference on Digital Libraries in 2020 (pp. 333-336).
