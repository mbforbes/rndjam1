# plan

Figuring out how to finish up this rndjam and move onto the next!

## restructuring

Current structure:
- Goal
- Running
- Data splits
- Viewing an image
- Data loading: CSV vs binary ("tensor")
- Naive regression to scalar
    - notation
    - OLS
    - ridge
    - lasso
- links
- acknowledgments

This structure is weird:
* big script to start
* no intro
* main content is buried in 3rd level headings

Restructure:
- Intro (incl brief intro and goal)
    - also note pytorch used here just as accelerated matrix lib, no autograd
    - consider adding takeaways up here:
        - surprisingly hard to find derivations for these "basic" algos all together
        - surprisingly tricky to do them if you really want to understand every step.
          Tricky bits: not regularizing bias terms, really tracking n throughout, people
          combining n and lambda, swapping column vs row vectors for derivatives
        - do multiclass regression instead of scalar regression
        - lasso requires coordinate descent for the weight sparsity property
        - surprisingly amazing: single regression code "just works" for multiclass
          regression
- Setup
- Data
    - Getting + splitting
    - viewing an image
    - Data loading
        - Note how this probably doesn't matter in practice
- Notation
- Linear: OLS
    - Naive regression to scalar (note + example)
    - (rest of OLS content)
    - Graphs + results
    - Multi-class regression to one-hots (note + example)
        - Consider updating a derivation (to show shapes work) and showing how code
          "just runs" (maybe shape annotations on code)
    - Graphs + results
- Linear: Ridge
    - Include section checking weight magnitudes vs OLS under a couple different
      regularization strengths
    - Don't spend much time on single regression to scalar results, focus on multiclass
      results and weight properties
    - (rest of ridge content)
- Linear: Lasso
    - Include discussion of weight 0s requiring coordinate descent as more main point
    - Include comparisons of weight 0s
    - (rest of lasso content)
- Logistic: Intro
- Logistic: MLE
- Logistic: Regularization (L2 definitely, maybe L1)
- Links
- Acknowledgments

Include somewhere: [MNIST scores reference](http://yann.lecun.com/exdb/mnist/) (test error %):
- linear: 12%
- deskew (preprocess) + linear: 8.5%
- KNN (L2): 5%
- deskew (preprocess) + KNN (L2): 2.4%
- Best CNNs: < 1%

Final runs on test (not that important)

Main other todos:
- check on logistic regression. does the sigmoid give us anything? if so, how much more
  math & code would it be?
    - Yes it does. This is worth another section(s).
    - https://math.stackexchange.com/questions/477207/derivative-of-cost-function-for-logistic-regression
    - https://sandipanweb.wordpress.com/2017/11/25/some-deep-learning-with-python-tensorflow-and-keras/
    - https://courses.cs.washington.edu/courses/cse599c1/13wi/slides/l2-regularization-online-perceptron.pdf
- Anything for updating to new pytorch?
- Make graphs a bit smaller or combine
- Balance text, equations, (code), and graphs
