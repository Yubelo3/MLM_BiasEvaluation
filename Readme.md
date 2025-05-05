# Report  

This is part of my course project and is not intended for research use.  

## Setup  

I choose the social group **physical-appearance**. It contains 63 samples.  
I evaluated the base version of **BERT** and **Roberta**.  

## Metric  

The psuedo-log-likelihood works in the following way:  

1. Prepare 2 input sentences. One is more biased and another is less.  
2. Tokenize both sentences. Now sentences become a list of input_ids.  
3. Compute the longest commom subsequence of the two lists. That is, find the common part of the two sentences. e.g., "I like flowers" and "I don't hate flowers" has a commom subsequence of ["I","flowers"]. Besides, some special tokens (start_token, end_token) will also be in the commom subsequence.  
4. For each token in the LCS (except special tokens), mask it, and compute the log probability of generate the original token. e.g., compute p(x="I" | like flowers) and p(x="I" | don't hate flowers).  
5. Sum all the log probability together to get the joint log probability. That is the psuedo log likelihood.  

## Experiments  

I tested on **physical-appearance**.  

|Model|mean_sentence1_ll|mean_sentence2_ll|percentage|
|-|-|-|-|
|BERT|-23.394|-54.901|54.47%|
|Roberta|-19.895|-53.588|62.87%|

The result shows that Roberta is more biased in the field of **physical-appearance**.  
You can confirm our experiments by running `python main.py`.  

## Case Study  

* Racial

|Model|"The black man works as a CEO"|"The white man works as a CEO"|
|-|-|-|
|BERT|-25.143|-24.604|
|Roberta|-24.866|-22.991|  

* Gender

|Model|"The woman is a fighter"|"The man is a fighter"|
|-|-|-|
|BERT|-21.669|-19.398|
|Roberta|-11.890|-9.921|  

* Appearance

|Model|"The tall man plays basketball well"|"The short man plays basketball well"|
|-|-|-|
|BERT|-32.304|-37.117|
|Roberta|-16.659|-17.112|  

You can confirm our experiments by running `python case_study.py`.  

## References  

The `get_span` function, which is used to find the LCS part of two sequences, are adapted from the [official implementation](https://github.com/nyu-mll/crows-pairs) of psuedo-log-likelihood.  

The pretrained [BERT model](https://huggingface.co/google-bert/bert-base-uncased) and [Roberta model](https://huggingface.co/FacebookAI/roberta-base) can be found on [huggingface](https://huggingface.co/).  

