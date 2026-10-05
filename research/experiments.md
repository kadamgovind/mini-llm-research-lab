# Experiments

## Experiment 001 — Zero-Probability Problem and Additive Smoothing

### Objective

Demonstrate the zero-probability problem in a maximum-likelihood
bigram language model and verify mathematically and experimentally
that additive smoothing assigns non-zero probability to unseen
bigrams.

### Research Question

What happens when a bigram has never appeared in the training corpus?

For a maximum-likelihood bigram model:

\[
P(w_i \mid w_{i-1})=
\frac{C(w_{i-1},w_i)}
{C(w_{i-1})}
\]

If:

\[
C(w_{i-1},w_i)=0
\]

then:

\[
P(w_i \mid w_{i-1})=0
\]

This creates the zero-probability problem.

## Experimental Corpus

The experiment uses the following token sequence:

```text
the dog the cat the dog
