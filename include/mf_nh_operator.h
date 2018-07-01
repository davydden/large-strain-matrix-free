#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <material.h>

using namespace dealii;

  /**
   * Large strain Neo-Hook tangent operator.
   *
   * Follow https://github.com/dealii/dealii/blob/master/tests/matrix_free/step-37.cc
   */
  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  class NeoHookOperator : public Subscriptor
  {
  public:
    NeoHookOperator ();

    using size_type = typename LinearAlgebra::distributed::Vector<number>::size_type;

    void clear();

    void initialize(std::shared_ptr<const MatrixFree<dim,number>> data_current,
                    std::shared_ptr<const MatrixFree<dim,number>> data_reference,
                    LinearAlgebra::distributed::Vector<number> &displacement);

    void set_material(std::shared_ptr<Material_Compressible_Neo_Hook_One_Field<dim,VectorizedArray<number>>> material);

    void compute_diagonal();

    unsigned int m () const;
    unsigned int n () const;

    void vmult (LinearAlgebra::distributed::Vector<number> &dst,
                const LinearAlgebra::distributed::Vector<number> &src) const;

    void Tvmult (LinearAlgebra::distributed::Vector<number> &dst,
                 const LinearAlgebra::distributed::Vector<number> &src) const;
    void vmult_add (LinearAlgebra::distributed::Vector<number> &dst,
                    const LinearAlgebra::distributed::Vector<number> &src) const;
    void Tvmult_add (LinearAlgebra::distributed::Vector<number> &dst,
                     const LinearAlgebra::distributed::Vector<number> &src) const;

    number el (const unsigned int row,
               const unsigned int col) const;

    void precondition_Jacobi(LinearAlgebra::distributed::Vector<number> &dst,
                             const LinearAlgebra::distributed::Vector<number> &src,
                             const number omega) const;

    const std::shared_ptr<DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>> get_matrix_diagonal_inverse() const
    {
      return inverse_diagonal_entries;
    }

    /**
     * Cache a few things for the current displacement.
     */
    void cache();

  private:

    /**
     * Apply operator on a range of cells.
     */
    void local_apply_cell (const MatrixFree<dim,number>    &data,
                           LinearAlgebra::distributed::Vector<number>                      &dst,
                           const LinearAlgebra::distributed::Vector<number>                &src,
                           const std::pair<unsigned int,unsigned int> &cell_range) const;

    /**
     * Apply diagonal part of the operator on a cell range.
     */
    void local_diagonal_cell (const MatrixFree<dim,number> &data,
                              LinearAlgebra::distributed::Vector<number>                                   &dst,
                              const unsigned int &,
                              const std::pair<unsigned int,unsigned int>       &cell_range) const;

   /**
    * Perform operation on a cell. @p phi_current and @phi_current_s correspond to the deformed configuration
    * where @p phi_reference is for the current configuration.
    */
   void do_operation_on_cell(FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> &phi_current,
                             FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> &phi_current_s,
                             FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> &phi_reference,
                             const unsigned int cell) const;

    std::shared_ptr<const MatrixFree<dim,number>> data_current;
    std::shared_ptr<const MatrixFree<dim,number>> data_reference;

    const LinearAlgebra::distributed::Vector<number> *displacement;

    std::shared_ptr<Material_Compressible_Neo_Hook_One_Field<dim,VectorizedArray<number>>> material;

    std::shared_ptr<DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>>  inverse_diagonal_entries;
    std::shared_ptr<DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>>  diagonal_entries;

    Table<2,VectorizedArray<number>> cached_scalar;

    bool            diagonal_is_available;
  };



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::NeoHookOperator ()
    :
    Subscriptor(),
    diagonal_is_available(false)
  {}



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::precondition_Jacobi(LinearAlgebra::distributed::Vector<number> &dst,
                                            const LinearAlgebra::distributed::Vector<number> &src,
                                            const number omega) const
  {
    Assert(inverse_diagonal_entries.get() &&
           inverse_diagonal_entries->m() > 0, ExcNotInitialized());
    inverse_diagonal_entries->vmult(dst,src);
    dst *= omega;
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  unsigned int
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::m () const
  {
    return data_current->get_vector_partitioner()->size();
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  unsigned int
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::n () const
  {
    return data_current->get_vector_partitioner()->size();
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::clear ()
  {
    data_current.reset();
    data_reference.reset();
    diagonal_is_available = false;
    diagonal_entries.reset();
    inverse_diagonal_entries.reset();
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::initialize(
                    std::shared_ptr<const MatrixFree<dim,number>> data_current_,
                    std::shared_ptr<const MatrixFree<dim,number>> data_reference_,
                    LinearAlgebra::distributed::Vector<number> &displacement_)
  {
    data_current = data_current_;
    data_reference = data_reference_;
    displacement = &displacement_;


    const unsigned int n_cells = data_reference_->n_macro_cells();
    FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> phi (*data_reference_);
    cached_scalar.reinit(n_cells,phi.n_q_points);
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::cache()
  {
    const unsigned int n_cells = data_reference->n_macro_cells();

    FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> phi_reference(*data_reference);

    for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
        phi_reference.reinit(cell);
        phi_reference.read_dof_values_plain(*displacement);
        phi_reference.evaluate (false,true,false);

        if (material->formulation == 0)
        {
          for (unsigned int q=0; q<phi_reference.n_q_points; ++q)
            {
              const Tensor<2,dim,VectorizedArray<number>>         &grad_u = phi_reference.get_gradient(q);
              const Tensor<2,dim,VectorizedArray<number>>          F      = Physics::Elasticity::Kinematics::F(grad_u);
              const VectorizedArray<number>                        det_F  = determinant(F);

              for (unsigned int i = 0;
                   i < data_current->n_components_filled(cell);
                   ++i)
                Assert(det_F[i] > 0,
                       ExcMessage(
                         "det_F[" + std::to_string(i) +
                         "] is not positive: " + std::to_string(det_F[i])));

              cached_scalar(cell,q) = std::pow(det_F,number(-1.0/dim));
            }
        }
        else if (material->formulation == 1)
        {
          for (unsigned int q=0; q<phi_reference.n_q_points; ++q)
            {
              const Tensor<2,dim,VectorizedArray<number>>         &grad_u = phi_reference.get_gradient(q);
              const Tensor<2,dim,VectorizedArray<number>>          F      = Physics::Elasticity::Kinematics::F(grad_u);
              const VectorizedArray<number>                        det_F  = determinant(F);

              for (unsigned int i = 0;
                   i < data_current->n_components_filled(cell);
                   ++i)
                Assert(det_F[i] > 0,
                       ExcMessage(
                         "det_F[" + std::to_string(i) +
                         "] is not positive: " + std::to_string(det_F[i])));

              cached_scalar(cell,q) = std::log(det_F);
            }
        }
        else
          AssertThrow(false, ExcMessage("Unknown material formulation"));
      }
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::set_material(std::shared_ptr<Material_Compressible_Neo_Hook_One_Field<dim,VectorizedArray<number>>> material_)
  {
    material = material_;
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::vmult (LinearAlgebra::distributed::Vector<number>       &dst,
                                                const LinearAlgebra::distributed::Vector<number> &src) const
  {
    dst = 0;
    vmult_add (dst, src);
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::Tvmult (LinearAlgebra::distributed::Vector<number>       &dst,
                                                 const LinearAlgebra::distributed::Vector<number> &src) const
  {
    dst = 0;
    vmult_add (dst,src);
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::Tvmult_add (LinearAlgebra::distributed::Vector<number>       &dst,
                                                     const LinearAlgebra::distributed::Vector<number> &src) const
  {
    vmult_add (dst,src);
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::vmult_add (LinearAlgebra::distributed::Vector<number>       &dst,
                                                    const LinearAlgebra::distributed::Vector<number> &src) const
  {
    // FIXME: use cell_loop, should work even though we need
    // both matrix-free data objects.
    Assert(data_current->n_macro_cells() == data_reference->n_macro_cells(),
           ExcInternalError());

    Assert (data_current->n_macro_cells() == data_reference->n_macro_cells(), ExcInternalError());

    // MatrixFree::cell_loop() is more complicated than a simple update_ghost_values() / compress(),
    // it loops on different cells (inner without ghosts and outer) in different order
    // and do update_ghost_values() and compress_start()/compress_finish() in between.
    // https://www.dealii.org/developer/doxygen/deal.II/matrix__free_8h_source.html#l00109

    // 1. make sure ghosts are updated
    src.update_ghost_values();

    // 2. loop over all locally owned cell blocks
    local_apply_cell(*data_current, dst, src,
                     std::make_pair<unsigned int,unsigned int>(0,data_current->n_macro_cells()));

    // 3. communicate results with MPI
    dst.compress(VectorOperation::add);

    // 4. constraints
    for (const auto dof : data_current->get_constrained_dofs())
      dst.local_element(dof) += src.local_element(dof);
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::local_apply_cell (
                           const MatrixFree<dim,number>    &/*data*/,
                           LinearAlgebra::distributed::Vector<number>                      &dst,
                           const LinearAlgebra::distributed::Vector<number>                &src,
                           const std::pair<unsigned int,unsigned int> &cell_range) const
  {
    // FIXME: I don't use data input, can this be bad?

    FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> phi_current  (*data_current);
    FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> phi_current_s(*data_current);
    FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> phi_reference(*data_reference);

    Assert (phi_current.n_q_points == phi_reference.n_q_points, ExcInternalError());

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        // initialize on this cell
        phi_current.reinit(cell);
        phi_current_s.reinit(cell);
        phi_reference.reinit(cell);

        // read-in total displacement and src vector and evaluate gradients
        phi_reference.read_dof_values_plain(*displacement);
        phi_current.  read_dof_values(src);
        phi_current_s.read_dof_values(src);

        do_operation_on_cell(phi_current,phi_current_s,phi_reference,cell);

        phi_current.distribute_local_to_global(dst);
        phi_current_s.distribute_local_to_global(dst);
      }
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::local_diagonal_cell (const MatrixFree<dim,number> &/*data*/,
                              LinearAlgebra::distributed::Vector<number>                                   &dst,
                              const unsigned int &,
                              const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    // FIXME: I don't use data input, can this be bad?
    const VectorizedArray<number> one = make_vectorized_array<number>(1.);
    const VectorizedArray<number> zero = make_vectorized_array<number>(0.);

    FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> phi_current  (*data_current);
    FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> phi_current_s(*data_current);
    FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> phi_reference(*data_reference);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        // initialize on this cell
        phi_current.reinit(cell);
        phi_current_s.reinit(cell);
        phi_reference.reinit(cell);

        // read-in total displacement.
        phi_reference.read_dof_values_plain(*displacement);

        // FIXME: although we override DoFs manually later, somehow
        // we still need to read some dummy here
        phi_current.read_dof_values(*displacement);
        phi_current_s.read_dof_values(*displacement);

        AlignedVector<VectorizedArray<number>> local_diagonal_vector(phi_current.dofs_per_component*phi_current.n_components);

        // Loop over all DoFs and set dof values to zero everywhere but i-th DoF.
        // With this input (instead of read_dof_values()) we do the action and store the
        // result in a diagonal vector
        for (unsigned int i=0; i<phi_current.dofs_per_component; ++i)
          for (unsigned int ic=0; ic<phi_current.n_components; ++ic)
            {
              for (unsigned int j=0; j<phi_current.dofs_per_component; ++j)
                for (unsigned int jc=0; jc<phi_current.n_components; ++jc)
                  {
                    const auto ind_j = j+jc*phi_current.dofs_per_component;
                    phi_current.begin_dof_values()  [ind_j] = zero;
                    phi_current_s.begin_dof_values()[ind_j] = zero;
                  }

              const auto ind_i = i+ic*phi_current.dofs_per_component;

              phi_current.begin_dof_values()  [ind_i] = one;
              phi_current_s.begin_dof_values()[ind_i] = one;

              do_operation_on_cell(phi_current,phi_current_s,phi_reference,cell);

              local_diagonal_vector[ind_i] = phi_current.begin_dof_values()[ind_i] +
                                             phi_current_s.begin_dof_values()[ind_i];
            }

        // Finally, in order to distribute diagonal, write it again into one of
        // FEEvaluations and do the standard distribute_local_to_global.
        // Note that here non-diagonal matrix elements are ignored and so the result is
        // not equivalent to matrix-based case when hanging nodes are present.
        // see Section 5.3 in Korman 2016, A time-space adaptive method for the Schrodinger equation, doi: 10.4208/cicp.101214.021015a
        // for a discussion.
        for (unsigned int i=0; i<phi_current.dofs_per_component; ++i)
          for (unsigned int ic=0; ic<phi_current.n_components; ++ic)
            {
              const auto ind_i = i+ic*phi_current.dofs_per_component;
              phi_current.begin_dof_values()[ind_i] = local_diagonal_vector[ind_i];
            }

        phi_current.distribute_local_to_global (dst);
      } // end of cell loop
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::do_operation_on_cell(
                             FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> &phi_current,
                             FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> &phi_current_s,
                             FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> &phi_reference,
                             const unsigned int cell) const
  {
    // make sure both MatrixFree objects use the same cells
    AssertDimension(data_current->n_components_filled(cell), data_reference->n_components_filled(cell));
    for (unsigned int i = 0; i < data_current->n_components_filled(cell); ++i)
      Assert (data_current->get_cell_iterator(cell,i) == data_reference->get_cell_iterator(cell,i),
              ExcMessage("Cell block " + std::to_string(cell) +
                         " element " + std::to_string(i) +
                         " does not match between two MatrixFree objects."
                         ));

    typedef VectorizedArray<number> NumberType;
    static constexpr number dim_f = dim;
    static constexpr number two_over_dim = 2.0/dim;
    const number kappa = material->kappa;
    const number c_1   = material->c_1;
    const number mu   = material->mu;
    const number lambda   = material->lambda;

    phi_reference.evaluate (false,true,false);
    phi_current.  evaluate (false,true,false);
    phi_current_s.evaluate (false,true,false);

    if (material->formulation == 0)
    {
      for (unsigned int q=0; q<phi_current.n_q_points; ++q)
        {
          // reference configuration:
          const Tensor<2,dim,NumberType>         &grad_u = phi_reference.get_gradient(q);
          const Tensor<2,dim,NumberType>          F      = Physics::Elasticity::Kinematics::F(grad_u);
          const NumberType                        det_F  = determinant(F);
          const Tensor<2,dim,NumberType>          F_bar  = F * cached_scalar(cell,q);
          const SymmetricTensor<2,dim,NumberType> b_bar  = Physics::Elasticity::Kinematics::b(F_bar);

          for (unsigned int i = 0; i < data_current->n_components_filled(cell); ++i)
            Assert (det_F[i] > 0, ExcMessage("det_F[" + std::to_string(i) + "] is not positive"));

          // current configuration
          const Tensor<2,dim,NumberType>          &grad_Nx_v      = phi_current.get_gradient(q);
          const SymmetricTensor<2,dim,NumberType> &symm_grad_Nx_v = phi_current.get_symmetric_gradient(q);

          // Next, determine the isochoric Kirchhoff stress
          // $\boldsymbol{\tau}_{\textrm{iso}} =
          // \mathcal{P}:\overline{\boldsymbol{\tau}}$
          const SymmetricTensor<2,dim,NumberType> tau_bar = b_bar * (2.0 * c_1);

          // trace of fictitious Kirchhoff stress
          // $\overline{\boldsymbol{\tau}}$:
          // 2.0 * c_1 * b_bar
          const NumberType tr_tau_bar = trace(tau_bar);

          const NumberType tr_tau_bar_dim = tr_tau_bar / dim_f;

          // Derivative of the volumetric free energy with respect to
          // $J$ return $\frac{\partial
          // \Psi_{\text{vol}}(J)}{\partial J}$
          const NumberType dPsi_vol_dJ = (kappa / 2.0) * (det_F - 1.0 / det_F);

          const NumberType dPsi_vol_dJ_J = dPsi_vol_dJ * det_F;

          const NumberType d2Psi_vol_dJ2 = ( (kappa / 2.0) * (1.0 + 1.0 / (det_F * det_F)));

          // Kirchoff stress:
          SymmetricTensor<2,dim,NumberType> tau;
          {
            tau = NumberType();
            // See Holzapfel p231 eq6.98 onwards

            // The following functions are used internally in determining the result
            // of some of the public functions above. The first one determines the
            // volumetric Kirchhoff stress $\boldsymbol{\tau}_{\textrm{vol}}$.
            // Note the difference in its definition when compared to step-44.
            for (unsigned int d = 0; d < dim; ++d)
              tau[d][d] = dPsi_vol_dJ_J - tr_tau_bar_dim;

            tau += tau_bar;
          }

          // material part of the action of tangent:
          // The action of the fourth-order material elasticity tensor in the spatial setting
          // on symmetric tensor.
          // $\mathfrak{c}$ is calculated from the SEF $\Psi$ as $ J
          // \mathfrak{c}_{ijkl} = F_{iA} F_{jB} \mathfrak{C}_{ABCD} F_{kC} F_{lD}$
          // where $ \mathfrak{C} = 4 \frac{\partial^2 \Psi(\mathbf{C})}{\partial
          // \mathbf{C} \partial \mathbf{C}}$
          SymmetricTensor<2,dim,VectorizedArray<number>> jc_part;
          {
            const NumberType tr = trace(symm_grad_Nx_v);

            SymmetricTensor<2,dim,NumberType> dev_src(symm_grad_Nx_v);
            for (unsigned int i = 0; i < dim; ++i)
              dev_src[i][i] -= tr/dim_f;

            // 1) The volumetric part of the tangent $J
            // \mathfrak{c}_\textrm{vol}$. Again, note the difference in its
            // definition when compared to step-44. The extra terms result from two
            // quantities in $\boldsymbol{\tau}_{\textrm{vol}}$ being dependent on
            // $\boldsymbol{F}$.
            // See Holzapfel p265

            // the term with the 4-th order symmetric tensor which gives symmetric
            // part of the tensor it acts on
            jc_part = symm_grad_Nx_v;
            jc_part*= - dPsi_vol_dJ_J*2.0;

            // term with IxI results in trace of the tensor times I
            const NumberType tmp = det_F * (dPsi_vol_dJ + det_F * d2Psi_vol_dJ2) * tr;
            for (unsigned int i = 0; i < dim; ++i)
              jc_part[i][i] += tmp;

            // 2) the isochoric part of the tangent $J
            // \mathfrak{c}_\textrm{iso}$:

            // The isochoric Kirchhoff stress
            // $\boldsymbol{\tau}_{\textrm{iso}} =
            // \mathcal{P}:\overline{\boldsymbol{\tau}}$:
            SymmetricTensor<2,dim,NumberType> tau_iso(tau_bar);
            for (unsigned int i = 0; i < dim; ++i)
              tau_iso[i][i] -= tr_tau_bar_dim;

            // term with deviatoric part of the tensor
            jc_part += (two_over_dim * tr_tau_bar) * dev_src;

            // term with tau_iso_x_I + I_x_tau_iso
            jc_part -= (two_over_dim * tr) * tau_iso;
            const NumberType tau_iso_src = tau_iso * symm_grad_Nx_v;
            for (unsigned int i = 0; i < dim; ++i)
              jc_part[i][i] -= two_over_dim * tau_iso_src;

            // c_bar==0 so we don't have a term with it.
          }

          const VectorizedArray<number> &JxW_current = phi_current.JxW(q);
          VectorizedArray<number>        JxW_scale   = phi_reference.JxW(q);
          for (unsigned int i = 0; i < data_current->n_components_filled(cell);
               ++i)
            {
              Assert(std::abs(JxW_current[i]) > 0., ExcInternalError());
              JxW_scale[i] *= 1. / JxW_current[i];
              // indirect check of consistency between MappingQEulerian in
              // MatrixFree data and displacement vector stored in this
              // operator.
              Assert(std::abs(JxW_scale[i] * det_F[i] - 1.) <
                       1000. * std::numeric_limits<number>::epsilon(),
                     ExcMessage(
                       std::to_string(i) + " out of " +
                       std::to_string(
                         VectorizedArray<number>::n_array_elements) +
                       ", filled " +
                       std::to_string(data_current->n_components_filled(cell)) +
                       " : " + std::to_string(det_F[i]) +
                       "!=" + std::to_string(1. / JxW_scale[i]) + " " +
                       std::to_string(std::abs(JxW_scale[i] * det_F[i] - 1.))));
            }

          // This is the $\mathsf{\mathbf{k}}_{\mathbf{u} \mathbf{u}}$
          // contribution. It comprises a material contribution, and a
          // geometrical stress contribution which is only added along
          // the local matrix diagonals:
          phi_current_s.submit_symmetric_gradient(
            jc_part * JxW_scale
            // Note: We need to integrate over the reference element, so the weights have to be adjusted
            ,q);

          // geometrical stress contribution
          // In index notation this tensor is $ [j e^{geo}]_{ijkl} = j \delta_{ik} \sigma^{tot}_{jl} = \delta_{ik} \tau^{tot}_{jl} $.
          // the product is actually  GradN * tau^T but due to symmetry of tau we can do GradN * tau
          const Tensor<2,dim,VectorizedArray<number>> tau_ns (tau);
          const Tensor<2,dim,VectorizedArray<number>> geo = grad_Nx_v * tau_ns;
          phi_current.submit_gradient(
            geo * JxW_scale
            // Note: We need to integrate over the reference element, so the weights have to be adjusted
            // phi_reference.JxW(q) / phi_current.JxW(q)
            ,q);

        } // end of the loop over quadrature points
    }
    else if (material->formulation == 1)
    {
      for (unsigned int q=0; q<phi_current.n_q_points; ++q)
        {
          // reference configuration:
          const Tensor<2,dim,NumberType>         &grad_u = phi_reference.get_gradient(q);
          const Tensor<2,dim,NumberType>          F      = Physics::Elasticity::Kinematics::F(grad_u);
          const SymmetricTensor<2,dim,NumberType> b      = Physics::Elasticity::Kinematics::b(F);

          const Tensor<2,dim,NumberType>          &grad_Nx_v      = phi_current.get_gradient(q);
          const SymmetricTensor<2,dim,NumberType> &symm_grad_Nx_v = phi_current.get_symmetric_gradient(q);

          SymmetricTensor<2,dim,NumberType> tau;
          {
            tau = mu*b;
            const NumberType tmp = mu - 2.0*lambda*cached_scalar(cell,q);
            for (unsigned int d = 0; d < dim; ++d)
              tau[d][d] -= tmp;
          }

          SymmetricTensor<2,dim,VectorizedArray<number>> jc_part;
          {
            jc_part = 2.0*(mu - 2.0*lambda*cached_scalar(cell,q))*symm_grad_Nx_v;
            const NumberType tmp = 2.0*lambda*trace(symm_grad_Nx_v);
            for (unsigned int i = 0; i < dim; ++i)
              jc_part[i][i] += tmp;
          }

          const NumberType               det_F       = determinant(F);
          const VectorizedArray<number> &JxW_current = phi_current.JxW(q);
          VectorizedArray<number>        JxW_scale   = phi_reference.JxW(q);
          for (unsigned int i = 0; i < data_current->n_components_filled(cell);
               ++i)
            {
              Assert(std::abs(JxW_current[i]) > 0., ExcInternalError());
              JxW_scale[i] *= 1. / JxW_current[i];
              // indirect check of consistency between MappingQEulerian in
              // MatrixFree data and displacement vector stored in this
              // operator.
              Assert(std::abs(JxW_scale[i] * det_F[i] - 1.) <
                       1000. * std::numeric_limits<number>::epsilon(),
                     ExcMessage(
                       std::to_string(i) + " out of " +
                       std::to_string(
                         VectorizedArray<number>::n_array_elements) +
                       ", filled " +
                       std::to_string(data_current->n_components_filled(cell)) +
                       " : " + std::to_string(det_F[i]) +
                       "!=" + std::to_string(1. / JxW_scale[i]) + " " +
                       std::to_string(std::abs(JxW_scale[i] * det_F[i] - 1.))));
            }

          phi_current_s.submit_symmetric_gradient(
            jc_part * JxW_scale,q);

          const Tensor<2,dim,VectorizedArray<number>> tau_ns (tau);
          const Tensor<2,dim,VectorizedArray<number>> geo = grad_Nx_v * tau_ns;
          phi_current.submit_gradient(
            geo * JxW_scale,q);
        }
    }
    else
      AssertThrow(false, ExcMessage("Unknown material formulation"));

    // actually do the contraction
    phi_current.integrate (false,true);
    phi_current_s.integrate (false,true);
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::
  compute_diagonal()
  {
    typedef LinearAlgebra::distributed::Vector<number> VectorType;

    inverse_diagonal_entries.reset(new DiagonalMatrix<VectorType>());
    diagonal_entries.reset(new DiagonalMatrix<VectorType>());
    VectorType &inverse_diagonal_vector = inverse_diagonal_entries->get_vector();
    VectorType &diagonal_vector         = diagonal_entries->get_vector();

    data_current->initialize_dof_vector(inverse_diagonal_vector);
    data_current->initialize_dof_vector(diagonal_vector);

    unsigned int dummy = 0;
    diagonal_vector = 0.;
    local_diagonal_cell(*data_current, diagonal_vector, dummy,
                     std::make_pair<unsigned int,unsigned int>(0,data_current->n_macro_cells()));
    diagonal_vector.compress(VectorOperation::add);

    // data_current->cell_loop (&NeoHookOperator::local_diagonal_cell,
    //                          this, diagonal_vector, dummy);

    // set_constrained_entries_to_one
    for (const auto dof : data_current->get_constrained_dofs())
      diagonal_vector.local_element(dof) = 1.;

    // calculate inverse:
    inverse_diagonal_vector = diagonal_vector;

    for (unsigned int i = 0; i < inverse_diagonal_vector.local_size(); ++i)
      {
        Assert(inverse_diagonal_vector.local_element(i) > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero or negative"));
        inverse_diagonal_vector.local_element(i) =
          1. / diagonal_vector.local_element(i);
      }

    inverse_diagonal_vector.update_ghost_values();
    diagonal_vector.update_ghost_values();

    diagonal_is_available = true;
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  number
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::el (const unsigned int row,
                                             const unsigned int col) const
  {
    Assert (row == col, ExcNotImplemented());
    (void)col;
    Assert (diagonal_is_available == true, ExcNotInitialized());
    return diagonal_entries->get_vector()(row);
  }
