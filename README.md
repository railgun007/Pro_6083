# Pro_6083
FRE 6083 Quantitative Method Final Project about Momentum Strategy

This paper studies L1 and L2 trend filtering methods. These methods are widely used in momentum strategies,
which correspond to an investment style based only on history of past prices. Various implementation of L1 filtering 
and L2 filtering are discussed at the beginning in order to detect some properties of noisy signals, which is using
penalty condition to obtain the filtered signal composed by a set of straight trends or steps. This penalty condition 
is implemented in a constrained least square problem and is represented by a regularization parameter 𝜆 which is 
estimated by a cross-validation procedure. Explicit applications to momentum strategies in S&P 500, SSE Composite 
Index and gold commodity are also discussed in detail with appropriate uses of the trend detection and configurations. 
The paper is concluded by listing some issues to consider when implementing a momentum strategy on index and commodity 
and some challenges we met and we want to improve in the future.
