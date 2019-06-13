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
- OLS
    - Naive regression to scalar (note + example)
    - (rest of OLS content)
    - Graphs + results
    - Multi-class regression to one-hots (note + example)
        - Consider updating a derivation (to show shapes work) and showing how code
          "just runs" (maybe shape annotations on code)
    - Graphs + results
- Ridge
    - Include section checking weight magnitudes vs OLS under a couple different
      regularization strengths
    - Don't spend much time on single regression to scalar results, focus on multiclass
      results and weight properties
    - (rest of ridge content)
- Lasso
    - Include discussion of weight 0s requiring coordinate descent as more main point
    - Include comparisons of weight 0s
    - (rest of lasso content)
- Links
- Acknowledgments


Main other todos:
- check on logistic regression. does the sigmoid give us anything? if so, how much more
  math & code would it be?
- Anything for updating to new pytorch?
- Make graphs a bit smaller or combine
