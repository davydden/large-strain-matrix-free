#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>

#include <config.h>
#include <material.h>

// mask for various ops in MF operator
enum class MFMask : unsigned
{
  Empty = 0,
  Zero = 1 << 0,                    // zero source vector
  MPI = 1 << 1,                     // do MPI communication (ghosts/compress)
  RW = 1 << 2,                      // read/write, constraints and reinit() of FEEvaluation
  SF = 1 << 3,                      // Local integration with sum factorization
  QD = 1 << 4,                      // Quadrature loop
  Default = (Zero | MPI | RW | SF| QD), // do all
  CellLoop = (RW | SF| QD),  // cell loop only
  RWSF     = (RW | SF)       // cell loop without QD
};

inline constexpr MFMask operator|(MFMask lhs, MFMask rhs)
{
  return static_cast<MFMask>(
    static_cast<std::underlying_type<MFMask>::type>(lhs) |
    static_cast<std::underlying_type<MFMask>::type>(rhs));
}

inline MFMask &operator|=(MFMask &f1, MFMask f2)
{
  f1 = f1 | f2;
  return f1;
}

inline constexpr bool operator&(MFMask f1, MFMask f2)
{
  return (static_cast<std::underlying_type<MFMask>::type>(f1) &
          static_cast<std::underlying_type<MFMask>::type>(f2)) != 0;
}

#ifdef WITH_LIKWID
#  include <likwid.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif


using namespace dealii;

template <typename Number>
void
adjust_ghost_range_if_necessary(
  const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner,
  const LinearAlgebra::distributed::Vector<Number> &        vec)
{
  if (vec.get_partitioner().get() != partitioner.get())
    {
      LinearAlgebra::distributed::Vector<Number> copy(vec);
      const_cast<LinearAlgebra::distributed::Vector<Number> &>(vec).reinit(
        partitioner);
      const_cast<LinearAlgebra::distributed::Vector<Number> &>(vec)
        .copy_locally_owned_data_from(copy);
    }
}


/**
 * Large strain Neo-Hook tangent operator.
 *
 * Follow
 * https://github.com/dealii/dealii/blob/master/tests/matrix_free/step-37.cc
 */
template <int dim, int fe_degree, int n_q_points_1d, typename number>
class NeoHookOperator : public Subscriptor
{
public:
  NeoHookOperator();

  using size_type =
    typename LinearAlgebra::distributed::Vector<number>::size_type;

  void
  clear();

  void
  initialize(std::shared_ptr<const MatrixFree<dim, number>>    data_current,
             std::shared_ptr<const MatrixFree<dim, number>>    data_reference,
             const LinearAlgebra::distributed::Vector<number> &displacement,
             const std::string caching);

  void
  set_material(
    std::shared_ptr<
      Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArray<number>>>
      material,
    std::shared_ptr<
      Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArray<number>>>
      material_inclusion);

  void
  compute_diagonal();

  unsigned int
  m() const;
  unsigned int
  n() const;

  template <MFMask mask = MFMask::Default>
  void
  vmult(LinearAlgebra::distributed::Vector<number> &      dst,
        const LinearAlgebra::distributed::Vector<number> &src) const;

  void
  Tvmult(LinearAlgebra::distributed::Vector<number> &      dst,
         const LinearAlgebra::distributed::Vector<number> &src) const;
  template <MFMask mask = MFMask::Default>
  void
  vmult_add(LinearAlgebra::distributed::Vector<number> &      dst,
            const LinearAlgebra::distributed::Vector<number> &src) const;
  void
  Tvmult_add(LinearAlgebra::distributed::Vector<number> &      dst,
             const LinearAlgebra::distributed::Vector<number> &src) const;

  number
  el(const unsigned int row, const unsigned int col) const;

  void
  precondition_Jacobi(LinearAlgebra::distributed::Vector<number> &      dst,
                      const LinearAlgebra::distributed::Vector<number> &src,
                      const number omega) const;

  const std::shared_ptr<
    DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>>
  get_matrix_diagonal_inverse() const
  {
    return inverse_diagonal_entries;
  }

  /**
   * Cache a few things for the current displacement.
   */
  void
  cache();

  /**
   * Cache memory consumption by this class.
   */
  std::size_t memory_consumption() const;

private:
  /**
   * Apply operator on a range of cells.
   */
  template <MFMask mask = MFMask::Default>
  void
  local_apply_cell(
    const MatrixFree<dim, number> &                   data,
    LinearAlgebra::distributed::Vector<number> &      dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const;

  /**
   * Apply diagonal part of the operator on a cell range.
   */
  void
  local_diagonal_cell(
    const MatrixFree<dim, number> &             data,
    LinearAlgebra::distributed::Vector<number> &dst,
    const unsigned int &,
    const std::pair<unsigned int, unsigned int> &cell_range) const;

  /**
   * Perform operation on a cell. @p phi_current corresponds to the deformed configuration
   * where @p phi_reference is for the current configuration.
   */
  template <MFMask mask = MFMask::Default>
  void
  do_operation_on_cell(
    FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number> &phi_current,
    FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number> &phi_reference,
    const unsigned int                                        cell) const;

  std::shared_ptr<const MatrixFree<dim, number>> data_current;
  std::shared_ptr<const MatrixFree<dim, number>> data_reference;

  const LinearAlgebra::distributed::Vector<number> *displacement;

  std::shared_ptr<
    Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArray<number>>>
    material;

  std::shared_ptr<
    Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArray<number>>>
    material_inclusion;

  std::shared_ptr<DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>>
    inverse_diagonal_entries;
  std::shared_ptr<DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>>
    diagonal_entries;

  Table<2, VectorizedArray<number>>                 cached_scalar;
  Table<2, VectorizedArray<number>>                 cached_second_scalar;
  Table<2, Tensor<2, dim, VectorizedArray<number>>> cached_tensor2;
  Table<2, SymmetricTensor<4, dim, VectorizedArray<number>>> cached_tensor4;

  bool diagonal_is_available;

  enum class MFCaching
    {
     none,
     scalar,
     tensor2,
     tensor4
    };
  MFCaching mf_caching;

  Tensor<2, dim, VectorizedArray<number>> zero_t2;
  SymmetricTensor<2, dim, VectorizedArray<number>> zero_t2s;

};



template <int dim, int fe_degree, int n_q_points_1d, typename number>
std::size_t
NeoHookOperator<dim, fe_degree, n_q_points_1d, number>::memory_consumption()
  const
{
  return cached_scalar.memory_consumption() +
         cached_second_scalar.memory_consumption() +
         cached_tensor2.memory_consumption() +
         cached_tensor4.memory_consumption() +
         // matrix-free data:
         data_current->memory_consumption() +
         (mf_caching == MFCaching::scalar ? data_reference->memory_consumption() : 0);
  // note: do not include diagonals, we want to measure only memory needed for vmult
  // for performance analysis.
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
NeoHookOperator<dim, fe_degree, n_q_points_1d, number>::NeoHookOperator()
  : Subscriptor()
  , diagonal_is_available(false)
  , mf_caching(MFCaching::none)
{
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      {
        zero_t2[i][j] = make_vectorized_array<number>(0.);
        zero_t2s[i][j] = make_vectorized_array<number>(0.);
      }
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, number>::precondition_Jacobi(
  LinearAlgebra::distributed::Vector<number> &      dst,
  const LinearAlgebra::distributed::Vector<number> &src,
  const number                                      omega) const
{
  Assert(inverse_diagonal_entries.get() && inverse_diagonal_entries->m() > 0,
         ExcNotInitialized());
  inverse_diagonal_entries->vmult(dst, src);
  dst *= omega;
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
unsigned int
NeoHookOperator<dim, fe_degree, n_q_points_1d, number>::m() const
{
  return data_current->get_vector_partitioner()->size();
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
unsigned int
NeoHookOperator<dim, fe_degree, n_q_points_1d, number>::n() const
{
  return data_current->get_vector_partitioner()->size();
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, number>::clear()
{
  data_current.reset();
  data_reference.reset();
  diagonal_is_available = false;
  diagonal_entries.reset();
  inverse_diagonal_entries.reset();
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, number>::initialize(
  std::shared_ptr<const MatrixFree<dim, number>>    data_current_,
  std::shared_ptr<const MatrixFree<dim, number>>    data_reference_,
  const LinearAlgebra::distributed::Vector<number> &displacement_,
  const std::string caching)
{
  data_current   = data_current_;
  data_reference = data_reference_;
  displacement   = &displacement_;

  const unsigned int n_cells = data_reference_->n_macro_cells();
  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number> phi(
    *data_reference_);
  if (caching == "scalar")
    {
      mf_caching = MFCaching::scalar;
      cached_scalar.reinit(n_cells, phi.n_q_points);
    }
  else if (caching == "tensor2")
    {
      mf_caching = MFCaching::tensor2;
      cached_scalar.reinit(n_cells, phi.n_q_points);
      cached_second_scalar.reinit(n_cells, phi.n_q_points);
      cached_tensor2.reinit(n_cells, phi.n_q_points);
    }
  else if (caching == "tensor4")
    {
      mf_caching = MFCaching::tensor4;
      cached_tensor2.reinit(n_cells, phi.n_q_points);
      cached_tensor4.reinit(n_cells, phi.n_q_points);
    }
  else
    {
      mf_caching = MFCaching::none;
      AssertThrow(false, ExcMessage("Unknown caching"));
    }
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, number>::cache()
{
  const unsigned int n_cells = data_reference->n_macro_cells();

  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number> phi_reference(
    *data_reference);

  for (unsigned int cell = 0; cell < n_cells; ++cell)
    {
      const unsigned int material_id =
        data_current->get_cell_iterator(cell, 0)->material_id();
      const auto &cell_mat = (material_id == 0 ? material : material_inclusion);

      phi_reference.reinit(cell);
      phi_reference.read_dof_values_plain(*displacement);
      phi_reference.evaluate(false, true, false);

      if (cell_mat->formulation == 0)
        {
          for (unsigned int q = 0; q < phi_reference.n_q_points; ++q)
            {
              const Tensor<2, dim, VectorizedArray<number>> &grad_u =
                phi_reference.get_gradient(q);
              const Tensor<2, dim, VectorizedArray<number>> F =
                Physics::Elasticity::Kinematics::F(grad_u);
              const VectorizedArray<number> det_F = determinant(F);

              for (unsigned int i = 0;
                   i < data_current->n_components_filled(cell);
                   ++i)
                Assert(det_F[i] > 0,
                       ExcMessage(
                         "det_F[" + std::to_string(i) +
                         "] is not positive: " + std::to_string(det_F[i])));

              cached_scalar(cell, q) = std::pow(det_F, number(-1.0 / dim));
            }
        }
      else if (cell_mat->formulation == 1)
        {
          for (unsigned int q = 0; q < phi_reference.n_q_points; ++q)
            {
              const Tensor<2, dim, VectorizedArray<number>> &grad_u =
                phi_reference.get_gradient(q);
              const Tensor<2, dim, VectorizedArray<number>> F =
                Physics::Elasticity::Kinematics::F(grad_u);
              const VectorizedArray<number> det_F = determinant(F);

              for (unsigned int i = 0;
                   i < data_current->n_components_filled(cell);
                   ++i)
                Assert(det_F[i] > 0,
                       ExcMessage(
                         "det_F[" + std::to_string(i) +
                         "] is not positive: " + std::to_string(det_F[i])));

              const VectorizedArray<number> scalar =
                cell_mat->mu - 2.0 * cell_mat->lambda * std::log(det_F);
              if (mf_caching == MFCaching::scalar)
                {
                  cached_scalar(cell, q) = scalar;
                }
              else if (mf_caching == MFCaching::tensor2)
                {
                  cached_scalar(cell, q)        = scalar * 2. / det_F;
                  cached_second_scalar(cell, q) = 2 * cell_mat->lambda / det_F;

                  SymmetricTensor<2, dim, VectorizedArray<number>> tau;
                  {
                    tau = cell_mat->mu * Physics::Elasticity::Kinematics::b(F);
                    for (unsigned int d = 0; d < dim; ++d)
                      tau[d][d] -= scalar;
                  }
                  cached_tensor2(cell, q) = tau / det_F;
                }
              else if (mf_caching == MFCaching::tensor4)
                {
                  SymmetricTensor<2, dim, VectorizedArray<number>> tau;
                  {
                    tau = cell_mat->mu * Physics::Elasticity::Kinematics::b(F);
                    for (unsigned int d = 0; d < dim; ++d)
                      tau[d][d] -= scalar;
                  }
                  cached_tensor2(cell, q) = tau / det_F;
                  cached_tensor4(cell, q) =
                    (scalar * 2. / det_F) *
                      Physics::Elasticity::StandardTensors<dim>::S +
                    (cell_mat->lambda * 2. / det_F) *
                      Physics::Elasticity::StandardTensors<dim>::IxI;
                }
              else
                {
                  AssertThrow(false, ExcMessage("Unknown caching"));
                }
            }
        }
      else
        AssertThrow(false, ExcMessage("Unknown material formulation"));
    }
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, number>::set_material(
  std::shared_ptr<
    Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArray<number>>>
    material_,
  std::shared_ptr<
    Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArray<number>>>
    material_inclusion_)
{
  material           = material_;
  material_inclusion = material_inclusion_;
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
template <MFMask mask>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, number>::vmult(
  LinearAlgebra::distributed::Vector<number> &      dst,
  const LinearAlgebra::distributed::Vector<number> &src) const
{
  if (mask & MFMask::Zero)
    dst = 0;
  vmult_add<mask>(dst, src);
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, number>::Tvmult(
  LinearAlgebra::distributed::Vector<number> &      dst,
  const LinearAlgebra::distributed::Vector<number> &src) const
{
  dst = 0;
  vmult_add(dst, src);
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, number>::Tvmult_add(
  LinearAlgebra::distributed::Vector<number> &      dst,
  const LinearAlgebra::distributed::Vector<number> &src) const
{
  vmult_add(dst, src);
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
template <MFMask mask>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, number>::vmult_add(
  LinearAlgebra::distributed::Vector<number> &      dst,
  const LinearAlgebra::distributed::Vector<number> &src) const
{
  const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner =
    data_current->get_vector_partitioner();

  Assert(partitioner->is_globally_compatible(
           *data_reference->get_vector_partitioner().get()),
         ExcMessage("Current and reference partitioners are incompatible"));

  if (mask & MFMask::MPI)
    {
      adjust_ghost_range_if_necessary(partitioner, dst);
      adjust_ghost_range_if_necessary(partitioner, src);
    }

  // FIXME: use cell_loop, should work even though we need
  // both matrix-free data objects.
  Assert(data_current->n_macro_cells() == data_reference->n_macro_cells(),
         ExcInternalError());

  // MatrixFree::cell_loop() is more complicated than a simple
  // update_ghost_values() / compress(), it loops on different cells (inner
  // without ghosts and outer) in different order and do update_ghost_values()
  // and compress_start()/compress_finish() in between.
  // https://www.dealii.org/developer/doxygen/deal.II/matrix__free_8h_source.html#l00109

  // 1. make sure ghosts are updated
  if (mask & MFMask::MPI)
    src.update_ghost_values();

  // 2. loop over all locally owned cell blocks
  local_apply_cell<mask>(*data_current,
                   dst,
                   src,
                   std::make_pair<unsigned int, unsigned int>(
                     0, data_current->n_macro_cells()));

  // 3. communicate results with MPI
  if (mask & MFMask::MPI)
    dst.compress(VectorOperation::add);

  // 4. constraints
  if (mask & MFMask::RW)
    for (const auto dof : data_current->get_constrained_dofs())
      dst.local_element(dof) += src.local_element(dof);
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
template <MFMask mask>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, number>::local_apply_cell(
  const MatrixFree<dim, number> & /*data*/,
  LinearAlgebra::distributed::Vector<number> &      dst,
  const LinearAlgebra::distributed::Vector<number> &src,
  const std::pair<unsigned int, unsigned int> &     cell_range) const
{
  // FIXME: I don't use data input, can this be bad?

  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number> phi_current(
    *data_current);
  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number> phi_reference(
    *data_reference);

  Assert(phi_current.n_q_points == phi_reference.n_q_points,
         ExcInternalError());

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // initialize on this cell
#if defined(WITH_LIKWID) && defined(WITH_BREAKDOWN)
      LIKWID_MARKER_START("vmult_reinit_read_write");
#endif
      if (mask & MFMask::RW)
        {
          phi_current.reinit(cell);

          // read-in total displacement and src vector and evaluate gradients
          phi_current.read_dof_values(src);

          if (mf_caching == MFCaching::scalar)
            {
              phi_reference.reinit(cell);
              phi_reference.read_dof_values_plain(*displacement);
            }
        }

#if defined(WITH_LIKWID) && defined(WITH_BREAKDOWN)
      LIKWID_MARKER_STOP("vmult_reinit_read_write");
#endif

      do_operation_on_cell<mask>(phi_current, phi_reference, cell);

#if defined(WITH_LIKWID) && defined(WITH_BREAKDOWN)
      LIKWID_MARKER_START("vmult_reinit_read_write");
#endif

      if (mask & MFMask::RW)
        {
          phi_current.distribute_local_to_global(dst);
        }
#if defined(WITH_LIKWID) && defined(WITH_BREAKDOWN)
      LIKWID_MARKER_STOP("vmult_reinit_read_write");
#endif

    }
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, number>::local_diagonal_cell(
  const MatrixFree<dim, number> & /*data*/,
  LinearAlgebra::distributed::Vector<number> &dst,
  const unsigned int &,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  // FIXME: I don't use data input, can this be bad?
  const VectorizedArray<number> one  = make_vectorized_array<number>(1.);
  const VectorizedArray<number> zero = make_vectorized_array<number>(0.);

  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number> phi_current(
    *data_current);
  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number> phi_reference(
    *data_reference);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // initialize on this cell
      phi_current.reinit(cell);
      phi_reference.reinit(cell);

      // read-in total displacement.
      phi_reference.read_dof_values_plain(*displacement);

      // FIXME: although we override DoFs manually later, somehow
      // we still need to read some dummy here
      phi_current.read_dof_values(*displacement);

      AlignedVector<VectorizedArray<number>> local_diagonal_vector(
        phi_current.dofs_per_component * phi_current.n_components);

      // Loop over all DoFs and set dof values to zero everywhere but i-th DoF.
      // With this input (instead of read_dof_values()) we do the action and
      // store the result in a diagonal vector
      for (unsigned int i = 0; i < phi_current.dofs_per_component; ++i)
        for (unsigned int ic = 0; ic < phi_current.n_components; ++ic)
          {
            for (unsigned int j = 0; j < phi_current.dofs_per_component; ++j)
              for (unsigned int jc = 0; jc < phi_current.n_components; ++jc)
                {
                  const auto ind_j = j + jc * phi_current.dofs_per_component;
                  phi_current.begin_dof_values()[ind_j]   = zero;
                }

            const auto ind_i = i + ic * phi_current.dofs_per_component;

            phi_current.begin_dof_values()[ind_i]   = one;

            do_operation_on_cell(phi_current,
                                 phi_reference,
                                 cell);

            local_diagonal_vector[ind_i] =
              phi_current.begin_dof_values()[ind_i];
          }

      // Finally, in order to distribute diagonal, write it again into one of
      // FEEvaluations and do the standard distribute_local_to_global.
      // Note that here non-diagonal matrix elements are ignored and so the
      // result is not equivalent to matrix-based case when hanging nodes are
      // present. see Section 5.3 in Korman 2016, A time-space adaptive method
      // for the Schrodinger equation, doi: 10.4208/cicp.101214.021015a for a
      // discussion.
      for (unsigned int i = 0; i < phi_current.dofs_per_component; ++i)
        for (unsigned int ic = 0; ic < phi_current.n_components; ++ic)
          {
            const auto ind_i = i + ic * phi_current.dofs_per_component;
            phi_current.begin_dof_values()[ind_i] =
              local_diagonal_vector[ind_i];
          }

      phi_current.distribute_local_to_global(dst);
    } // end of cell loop
}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
template <MFMask mask>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, number>::do_operation_on_cell(
  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number> &phi_current,
  FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number> &phi_reference,
  const unsigned int                                        cell) const
{
  const unsigned int material_id =
    data_current->get_cell_iterator(cell, 0)->material_id();
  const auto &cell_mat = (material_id == 0 ? material : material_inclusion);

  // make sure all filled cells have the same ID
#ifdef DEBUG
  for (unsigned int i = 1; i < data_current->n_components_filled(cell); ++i)
    {
      const unsigned int other_id =
        data_current->get_cell_iterator(cell, i)->material_id();
      Assert(other_id == material_id,
             ExcMessage("Cell block " + std::to_string(cell) + " element " +
                        std::to_string(i) + " has material ID " +
                        std::to_string(other_id) +
                        ", different form 0-th element " +
                        std::to_string(material_id)));
    }
#endif

  // make sure both MatrixFree objects use the same cells
  AssertDimension(data_current->n_components_filled(cell),
                  data_reference->n_components_filled(cell));
  for (unsigned int i = 0; i < data_current->n_components_filled(cell); ++i)
    Assert(data_current->get_cell_iterator(cell, i) ==
             data_reference->get_cell_iterator(cell, i),
           ExcMessage("Cell block " + std::to_string(cell) + " element " +
                      std::to_string(i) +
                      " does not match between two MatrixFree objects."));

  using NumberType                     = VectorizedArray<number>;
  static constexpr number dim_f        = dim;
  static constexpr number two_over_dim = 2.0 / dim;
  const number            kappa        = cell_mat->kappa;
  const number            c_1          = cell_mat->c_1;
  const number            mu           = cell_mat->mu;
  const number            lambda       = cell_mat->lambda;

#if defined(WITH_LIKWID) && defined(WITH_BREAKDOWN)
  LIKWID_MARKER_START("vmult_sum_factorization");
#endif

  if (mask & MFMask::SF)
    {
      if (mf_caching == MFCaching::scalar)
        phi_reference.evaluate(false, true, false);

      phi_current.evaluate(false, true, false);
    }

#if defined(WITH_LIKWID) && defined(WITH_BREAKDOWN)
  LIKWID_MARKER_STOP("vmult_sum_factorization");
#endif

#if defined(WITH_LIKWID) && defined(WITH_BREAKDOWN)
  LIKWID_MARKER_START("vmult_quadrature_loop");
#endif

  if (mask & MFMask::QD)
    {
      if (cell_mat->formulation == 0)
        {
          for (unsigned int q = 0; q < phi_current.n_q_points; ++q)
            {
              // reference configuration:
              const Tensor<2, dim, NumberType> &grad_u =
                phi_reference.get_gradient(q);
              const Tensor<2, dim, NumberType> F =
                Physics::Elasticity::Kinematics::F(grad_u);
              const NumberType                 det_F = determinant(F);
              const Tensor<2, dim, NumberType> F_bar = F * cached_scalar(cell, q);
              const SymmetricTensor<2, dim, NumberType> b_bar =
                Physics::Elasticity::Kinematics::b(F_bar);

              Assert(cached_scalar(cell, q) == std::pow(det_F, number(-1.0 / dim)),
                    ExcMessage("Cached scalar and det_F do not match"));

              for (unsigned int i = 0; i < data_current->n_components_filled(cell);
                  ++i)
                Assert(det_F[i] > 0,
                      ExcMessage("det_F[" + std::to_string(i) +
                                  "] is not positive"));

              // current configuration
              const Tensor<2, dim, NumberType> grad_Nx_v =
                phi_current.get_gradient(q);
              const SymmetricTensor<2, dim, NumberType> symm_grad_Nx_v =
                symmetrize(grad_Nx_v);

              // Next, determine the isochoric Kirchhoff stress
              // $\boldsymbol{\tau}_{\textrm{iso}} =
              // \mathcal{P}:\overline{\boldsymbol{\tau}}$
              const SymmetricTensor<2, dim, NumberType> tau_bar =
                b_bar * (2.0 * c_1);

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

              const NumberType d2Psi_vol_dJ2 =
                ((kappa / 2.0) * (1.0 + 1.0 / (det_F * det_F)));

              // Kirchoff stress:
              SymmetricTensor<2, dim, NumberType> tau;
              {
                tau = NumberType();
                // See Holzapfel p231 eq6.98 onwards

                // The following functions are used internally in determining the
                // result of some of the public functions above. The first one
                // determines the volumetric Kirchhoff stress
                // $\boldsymbol{\tau}_{\textrm{vol}}$. Note the difference in its
                // definition when compared to step-44.
                for (unsigned int d = 0; d < dim; ++d)
                  tau[d][d] = dPsi_vol_dJ_J - tr_tau_bar_dim;

                tau += tau_bar;
              }

              // material part of the action of tangent:
              // The action of the fourth-order material elasticity tensor in the
              // spatial setting on symmetric tensor.
              // $\mathfrak{c}$ is calculated from the SEF $\Psi$ as $ J
              // \mathfrak{c}_{ijkl} = F_{iA} F_{jB} \mathfrak{C}_{ABCD} F_{kC}
              // F_{lD}$ where $ \mathfrak{C} = 4 \frac{\partial^2
              // \Psi(\mathbf{C})}{\partial \mathbf{C} \partial \mathbf{C}}$
              SymmetricTensor<2, dim, VectorizedArray<number>> jc_part;
              {
                const NumberType tr = trace(symm_grad_Nx_v);

                SymmetricTensor<2, dim, NumberType> dev_src(symm_grad_Nx_v);
                for (unsigned int i = 0; i < dim; ++i)
                  dev_src[i][i] -= tr / dim_f;

                // 1) The volumetric part of the tangent $J
                // \mathfrak{c}_\textrm{vol}$. Again, note the difference in its
                // definition when compared to step-44. The extra terms result from
                // two quantities in $\boldsymbol{\tau}_{\textrm{vol}}$ being
                // dependent on
                // $\boldsymbol{F}$.
                // See Holzapfel p265

                // the term with the 4-th order symmetric tensor which gives
                // symmetric part of the tensor it acts on
                jc_part = symm_grad_Nx_v;
                jc_part *= -dPsi_vol_dJ_J * 2.0;

                // term with IxI results in trace of the tensor times I
                const NumberType tmp =
                  det_F * (dPsi_vol_dJ + det_F * d2Psi_vol_dJ2) * tr;
                for (unsigned int i = 0; i < dim; ++i)
                  jc_part[i][i] += tmp;

                // 2) the isochoric part of the tangent $J
                // \mathfrak{c}_\textrm{iso}$:

                // The isochoric Kirchhoff stress
                // $\boldsymbol{\tau}_{\textrm{iso}} =
                // \mathcal{P}:\overline{\boldsymbol{\tau}}$:
                SymmetricTensor<2, dim, NumberType> tau_iso(tau_bar);
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

    #ifdef DEBUG
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
    #endif

              // jc_part is the $\mathsf{\mathbf{k}}_{\mathbf{u} \mathbf{u}}$
              // contribution. It comprises a material contribution, and a
              // geometrical stress contribution which is only added along
              // the local matrix diagonals:

              // geometrical stress contribution
              // In index notation this tensor is $ [j e^{geo}]_{ijkl} = j
              // \delta_{ik} \sigma^{tot}_{jl} = \delta_{ik} \tau^{tot}_{jl} $. the
              // product is actually  GradN * tau^T but due to symmetry of tau we
              // can do GradN * tau
              const Tensor<2, dim, VectorizedArray<number>> tau_ns(tau);
              const Tensor<2, dim, VectorizedArray<number>> geo =
                grad_Nx_v * tau_ns;
              phi_current.submit_gradient(
                (Tensor<2,dim,VectorizedArray<number>>(jc_part)+
                 geo) / det_F
                // Note: We need to integrate over the reference element,
                // thus we divide by det_F so that FEEvaluation with
                // mapping does the right thing.
                ,
                q);

            } // end of the loop over quadrature points
        }
      else if (cell_mat->formulation == 1 && mf_caching == MFCaching::scalar)
        // the least amount of cache and the most calculations
        {
          for (unsigned int q = 0; q < phi_current.n_q_points; ++q)
            {
              // reference configuration:
              const Tensor<2, dim, NumberType> grad_u =
                phi_reference.get_gradient(q);
              const Tensor<2, dim, NumberType> F =
                Physics::Elasticity::Kinematics::F(grad_u);
              const SymmetricTensor<2, dim, NumberType> b =
                Physics::Elasticity::Kinematics::b(F);

              const Tensor<2, dim, NumberType> grad_Nx_v =
                phi_current.get_gradient(q);
              const SymmetricTensor<2, dim, NumberType> symm_grad_Nx_v =
                symmetrize(grad_Nx_v);

              SymmetricTensor<2, dim, NumberType> tau;
              {
                tau = mu * b;
                for (unsigned int d = 0; d < dim; ++d)
                  tau[d][d] -= cached_scalar(cell, q);
              }

              SymmetricTensor<2, dim, VectorizedArray<number>> jc_part;
              {
                jc_part = number(2.0) * cached_scalar(cell, q) * symm_grad_Nx_v;
                const NumberType tmp = number(2.0) * lambda * trace(symm_grad_Nx_v);
                for (unsigned int i = 0; i < dim; ++i)
                  jc_part[i][i] += tmp;
              }

              const NumberType det_F = determinant(F);
              Assert(cached_scalar(cell, q) ==
                       (mu - number(2.0) * lambda * std::log(det_F)),
                    ExcMessage("Cached scalar and det_F do not match"));

    #ifdef DEBUG
              const VectorizedArray<number>  JxW_current = phi_current.JxW(q);
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
                          100000. * std::numeric_limits<number>::epsilon(),
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
    #endif

              const Tensor<2, dim, VectorizedArray<number>> tau_ns(tau);
              const Tensor<2, dim, VectorizedArray<number>> geo =
                grad_Nx_v * tau_ns;
              phi_current.submit_gradient((Tensor<2, dim, VectorizedArray<number>>(jc_part) +
                                           geo) / det_F, q);
            }
        }
      else if (cell_mat->formulation == 1 && mf_caching == MFCaching::tensor2)
        // moderate cache of two scalar + 2nd order tensor
        {
          for (unsigned int q = 0; q < phi_current.n_q_points; ++q)
            {
              const Tensor<2, dim, NumberType> grad_Nx_v =
                phi_current.get_gradient(q);
              const SymmetricTensor<2, dim, NumberType> symm_grad_Nx_v =
                symmetrize(grad_Nx_v);

              SymmetricTensor<2, dim, VectorizedArray<number>> jc_part;
              {
                jc_part = cached_scalar(cell, q) * symm_grad_Nx_v;
                const NumberType tmp =
                  cached_second_scalar(cell, q) * trace(symm_grad_Nx_v);
                for (unsigned int i = 0; i < dim; ++i)
                  jc_part[i][i] += tmp;
              }

              phi_current.submit_gradient(Tensor<2, dim, NumberType>(jc_part) +
                                          grad_Nx_v * cached_tensor2(cell, q), q);
            }
        }
      else if (cell_mat->formulation == 1 && mf_caching == MFCaching::tensor4)
        // maximum cache (2nd order  4th order sym)
        {
          for (unsigned int q = 0; q < phi_current.n_q_points; ++q)
            {
              const Tensor<2, dim, NumberType> &grad_Nx_v =
                phi_current.get_gradient(q);
              const SymmetricTensor<2, dim, NumberType> &symm_grad_Nx_v =
                symmetrize(grad_Nx_v);

              phi_current.submit_gradient(grad_Nx_v * cached_tensor2(cell, q) +
                                          Tensor<2,dim,NumberType>(cached_tensor4(cell, q) *
                                                                   symm_grad_Nx_v), q);
            }
        }
      else
        AssertThrow(false, ExcMessage("Unknown material formulation/caching"));

    } // quadrature loop compile-time enum
  else
    {
      // need to submit something.
      // for (unsigned int q = 0; q < phi_current.n_q_points; ++q)
      const unsigned int q = 0;
      phi_current.submit_gradient(zero_t2, q);
    }

#if defined(WITH_LIKWID) && defined(WITH_BREAKDOWN)
  LIKWID_MARKER_STOP("vmult_quadrature_loop");
#endif

  // actually do the contraction
#if defined(WITH_LIKWID) && defined(WITH_BREAKDOWN)
  LIKWID_MARKER_START("vmult_sum_factorization");
#endif

  if (mask & MFMask::SF)
    {
      phi_current.integrate(false, true);
    }
#if defined(WITH_LIKWID) && defined(WITH_BREAKDOWN)
  LIKWID_MARKER_STOP("vmult_sum_factorization");
#endif

}



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
NeoHookOperator<dim, fe_degree, n_q_points_1d, number>::compute_diagonal()
{
  typedef LinearAlgebra::distributed::Vector<number> VectorType;

  inverse_diagonal_entries.reset(new DiagonalMatrix<VectorType>());
  diagonal_entries.reset(new DiagonalMatrix<VectorType>());
  VectorType &inverse_diagonal_vector = inverse_diagonal_entries->get_vector();
  VectorType &diagonal_vector         = diagonal_entries->get_vector();

  data_current->initialize_dof_vector(inverse_diagonal_vector);
  data_current->initialize_dof_vector(diagonal_vector);

  unsigned int dummy = 0;
  diagonal_vector    = 0.;
  local_diagonal_cell(*data_current,
                      diagonal_vector,
                      dummy,
                      std::make_pair<unsigned int, unsigned int>(
                        0, data_current->n_macro_cells()));
  diagonal_vector.compress(VectorOperation::add);

  // data_current->cell_loop (&NeoHookOperator::local_diagonal_cell,
  //                          this, diagonal_vector, dummy);

  // set_constrained_entries_to_one
  Assert(data_current->get_constrained_dofs().size() ==
           data_reference->get_constrained_dofs().size(),
         ExcInternalError());
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
NeoHookOperator<dim, fe_degree, n_q_points_1d, number>::el(
  const unsigned int row,
  const unsigned int col) const
{
  Assert(row == col, ExcNotImplemented());
  (void)col;
  Assert(diagonal_is_available == true, ExcNotInitialized());
  return diagonal_entries->get_vector()(row);
}
