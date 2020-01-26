# A matrix-free approach for finite-strain hyperelastic problems using geometric multigrid

We thank the anonymous reviewers for the valuable comments and suggestions to the manuscript. We did our best to incorporate them into the revision. Below we state our response point-by-point.
To facilitate the review process, all significant changes are marked in red in the updated version of the manuscript.

----


## Referee: 1

> Minor editing is required in figures (axis titles).

We have re-examined axis titles in all figures. We did not find any typos, but increased the font size of some of them to make more readable.

> A uniform convention should be adopted for author names in the list of references.

The multiple first names were abbreviated the same way. This is the only non-uniformity we could find for the author names.

----

## Referee: 2

> In this paper, the authors develop a matrix-free method for nonlinear elasticity. The paper is very well written, both the methods (Alg. 1-3) and the numerical tests are convincing.

> Here are some minor remarks:

> 1. Eq. (4) on page 3:
>  a) The period "." is wrong. Please make a comma
>  b) And \forall \delta u reads strange wittout function space. I would write \forall \text{admissible directions } \delta u

We have fixed these problems. Note that we removed \forall \delta u because it is explained in the text that follows, "vanishes for all admissible directions \delta u which satisfy homogeneous Dirichlet boundary conditions".

> 2. To non-familiar Heidelberg-persons, IWR might be unknown. Please explain IWR on page 12.

We have replaced the system name "IWR" by "Westmere", which is the name of the CPU architecture and consistent with the use of "Cascade Lake" for the other system.

> 3. Section 6: Out of curiosity: how many nonlinear Newton steps are required to solve the problem?

With our settings it takes 4 Newton-Raphson iterations to converge within a load step for the 2D problem and 3 iterations for the 3D setup.

> 4. In the conclusions, I would specifically refer again to Alg. 1,2,3 and give some suggestions which Alg. 1 finally is prefererred and why.
> On page 16, just before Section 6.2, there are some suggestions. Are these the final conclusions w.r.t. the proposed Alg.?

If linearization of the chosen material model allows to efficiently implement the action of the material part of the fourth-order spatial tangent stiffness tensor on the second-order symmetric tensor, then Algorithm 2 is recommended.
For applications within general frameworks with different geometrically non-linear constitutive models, Algorithm 3 provides the necessary flexibility while still being considerably faster than matrix-based strategies for polynomial degrees higher than one.

Note that the higher arithmetic load for Algorithm 1 is not surprising as at each quadrature point we need to re-evaluate the Kirchhoff stress, and evaluate gradients with respect to both the referential and the current configuration.
Thus no advantage of the scalar caching can be deduced despite the higher arithmetic performance.

We have augmented Conclusion section to better convey this message.