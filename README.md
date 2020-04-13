# A matrix-free approach for finite-strain hyperelastic problems using geometric multigrid

This is the repository for the code associated with the paper 

D. Davydov, J-P. Pelteret, D. Arndt, M. Kronbichler, P. Steinmann.
A matrix‐free approach for finite‐strain hyperelastic problems using geometric multigrid. 
International Journal for Numerical Methods in Engineering, 2020; 1– 22.
DOI: [10.1002/nme.6336](https://doi.org/10.1002/nme.6336).
arXiv: [1904.13131](https://arxiv.org/abs/1904.13131)

If you use this work, or find it useful in general, the citation of the aforementioned article would be much appreciated by the authors.

## Abstract
This work investigates matrix‐free algorithms for problems in quasi‐static finite‐strain hyperelasticity. 
Iterative solvers with matrix‐free operator evaluation have emerged as an attractive alternative to sparse matrices in the fluid dynamics and wave propagation communities because they significantly reduce the memory traffic, the limiting factor in classical finite element solvers.
Specifically, we study different matrix‐free realizations of the finite element tangent operator and determine whether generalized methods of incorporating complex constitutive behavior might be feasible. 
In order to improve the convergence behavior of iterative solvers, we also propose a method by which to construct level tangent operators and employ them to define a geometric multigrid preconditioner. 
The performance of the matrix‐free operator and the geometric multigrid preconditioner is compared to the matrix‐based implementation with an algebraic multigrid (AMG) preconditioner on a single node for a representative numerical example of a heterogeneous hyperelastic material in two and three dimensions. 
We find that matrix‐free methods for finite‐strain solid mechanics are very promising, outperforming linear matrix‐based schemes by two to five times, and that it is possible to develop numerically efficient implementations that are independent of the hyperelastic constitutive law.
