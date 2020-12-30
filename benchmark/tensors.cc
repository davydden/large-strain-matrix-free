
// benchmark double dot product between symmetric tensors and single dot
// between two second order tensors

#include <deal.II/base/config.h>

#include <deal.II/base/revision.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/table.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/physics/elasticity/standard_tensors.h>

// we need to include our config to see if we build with LIKWID:
#include <config.h>
// we may as well include version:
#include <version.h>

#ifdef WITH_LIKWID
#  include <likwid.h>
#else
// if build without LIKWID, keep dummy definitions to avoid writing #ifdef...
// all over the place
#  define LIKWID_MARKER_INIT
#  define LIKWID_MARKER_THREADINIT
#  define LIKWID_MARKER_SWITCH
#  define LIKWID_MARKER_REGISTER(regionTag)
#  define LIKWID_MARKER_START(regionTag)
#  define LIKWID_MARKER_STOP(regionTag)
#  define LIKWID_MARKER_CLOSE
#  define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif


#if (!defined(__GNUC__) && !defined(__clang__)) || defined(__pnacl__) || \
  defined(__EMSCRIPTEN__)
#  define BENCHMARK_HAS_NO_INLINE_ASSEMBLY
#endif

// The DoNotOptimize(...) function can be used to prevent a value or
// expression from being optimized away by the compiler. This function is
// intended to add little to no overhead.
// See: https://youtu.be/nXaxk27zwlk?t=2441
#ifndef BENCHMARK_HAS_NO_INLINE_ASSEMBLY
template <class Tp>
inline __attribute__((always_inline)) void
DoNotOptimize(Tp const &value)
{
  asm volatile("" : : "r,m"(value) : "memory");
}

template <class Tp>
inline __attribute__((always_inline)) void
DoNotOptimize(Tp &value)
{
#  if defined(__clang__)
  asm volatile("" : "+r,m"(value) : : "memory");
#  else
  asm volatile("" : "+m,r"(value) : : "memory");
#  endif
}

// Force the compiler to flush pending writes to global memory. Acts as an
// effective read/write barrier
inline __attribute__((always_inline)) void
ClobberMemory()
{
  asm volatile("" : : : "memory");
}
#else
template <class Tp>
inline __attribute__((always_inline)) void
DoNotOptimize(Tp const &value)
{
  internal::UseCharPointer(&reinterpret_cast<char const volatile &>(value));
}
// FIXME Add ClobberMemory() for non-gnu and non-msvc compilers
#endif

using namespace dealii;
using namespace Cook_Membrane;

template <int dim = 2, int degree = 4, typename number = double>
void
test(const unsigned int n_cells = 90112 / VectorizedArray<number>::size())
{
  constexpr int n_q_points = Utilities::pow(degree + 1, dim);

  std::cout
    << "-----------------------------------------------------------------------------"
    << std::endl
#ifdef DEBUG
    << "--     . running in DEBUG mode" << std::endl;
#else
    << "--     . running in OPTIMIZED mode" << std::endl;
#endif

  std::cout << "--     . vectorization over " << VectorizedArray<number>::size()
            << " elements, VECTORIZATION_LEVEL="
            << DEAL_II_COMPILER_VECTORIZATION_LEVEL << std::endl;

  std::cout << "--     . version " << GIT_TAG << " (revision " << GIT_SHORTREV
            << " on branch " << GIT_BRANCH << ")" << std::endl;
  std::cout << "--     . deal.II " << DEAL_II_PACKAGE_VERSION << " (revision "
            << DEAL_II_GIT_SHORTREV << " on branch " << DEAL_II_GIT_BRANCH
            << ")" << std::endl;

  std::cout
    << "-----------------------------------------------------------------------------"
    << std::endl
    << std::endl;

  std::cout << "dim                = " << dim << std::endl
            << "degree             = " << degree << std::endl
            << "SIMD width         = " << VectorizedArray<number>::size()
            << std::endl
            << "n_cell_batches     = " << n_cells << std::endl
            << "n_q_points         = " << n_q_points << std::endl
            << "number of products = " << n_cells * n_q_points << std::endl
            << "multiplier         = "
            << 1000000. /
                 (n_cells * n_q_points) // Runtime * MFLOP/s * multiplier = FLOP
                                        // for a single operation
            << std::endl;

  Table<2, Tensor<2, dim, VectorizedArray<number>>>          cached_tensor2;
  Table<2, SymmetricTensor<2, dim, VectorizedArray<number>>> cached_tensor2sym;
  Table<2, SymmetricTensor<4, dim, VectorizedArray<number>>> cached_tensor4;

  cached_tensor2.reinit(n_cells, n_q_points);
  cached_tensor4.reinit(n_cells, n_q_points);
  cached_tensor2sym.reinit(n_cells, n_q_points);

  // set tensors to something (does not matter)
  const VectorizedArray<number> one  = make_vectorized_array(1.);
  const VectorizedArray<number> zero = make_vectorized_array(0.);

  for (unsigned int cell = 0; cell < n_cells; ++cell)
    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        cached_tensor2(cell, q) =
          one * Physics::Elasticity::StandardTensors<dim>::I;
        cached_tensor2sym(cell, q) =
          one * Physics::Elasticity::StandardTensors<dim>::I;
        cached_tensor4(cell, q) =
          one * Physics::Elasticity::StandardTensors<dim>::S +
          one * Physics::Elasticity::StandardTensors<dim>::IxI;
      }

  // now the actual test.
  // Note that with FEEvaluation we always ask for each cell and each quadrature
  // point gradients and symmetric gradients. Here define them once outside of
  // the double loop. This will change computational intensity (won't need to
  // fetch data), but as far as Tensor<>::operator* is concerned, we should get
  // the same accurate measurements
  const Tensor<2, dim, VectorizedArray<number>> grad_Nx_v =
    one * Physics::Elasticity::StandardTensors<dim>::I;
  const SymmetricTensor<2, dim, VectorizedArray<number>> symm_grad_Nx_v =
    one * Physics::Elasticity::StandardTensors<dim>::I;

  TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

  LIKWID_MARKER_INIT;

  // first test 4-th order tensor product.
  {
    timer.enter_subsection("tensor4");
    LIKWID_MARKER_START("tensor4");
    SymmetricTensor<2, dim, VectorizedArray<number>> res;
    for (unsigned int cell = 0; cell < n_cells; ++cell)
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          // the result which goes to FEEvaluation::submit_symmetric_gradient()
          DoNotOptimize(res = cached_tensor4(cell, q) * symm_grad_Nx_v);
        }
    LIKWID_MARKER_STOP("tensor4");
    timer.leave_subsection();
  }

  // now test 2-nd order tensors

  // OPTION1 : operator*
  {
    timer.enter_subsection("tensor2");
    LIKWID_MARKER_START("tensor2");
    Tensor<2, dim, VectorizedArray<number>> res;
    for (unsigned int cell = 0; cell < n_cells; ++cell)
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          // the results which goes to FEEvaluation::submit_gradient()
          DoNotOptimize(res = grad_Nx_v * cached_tensor2(cell, q));
        }
    LIKWID_MARKER_STOP("tensor2");
    timer.leave_subsection();
  }

  // OPTION2: contract<1,0>
  {
    timer.enter_subsection("tensor2_c10");
    LIKWID_MARKER_START("tensor2_c10");
    Tensor<2, dim, VectorizedArray<number>> res;
    for (unsigned int cell = 0; cell < n_cells; ++cell)
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          // the results which goes to FEEvaluation::submit_gradient()
          DoNotOptimize(res =
                          contract<1, 0>(grad_Nx_v, cached_tensor2(cell, q)));
        }
    LIKWID_MARKER_STOP("tensor2_c10");
    timer.leave_subsection();
  }

  // OPTION3: contract<1,1>
  {
    timer.enter_subsection("tensor2_c11");
    LIKWID_MARKER_START("tensor2_c11");
    Tensor<2, dim, VectorizedArray<number>> res;
    for (unsigned int cell = 0; cell < n_cells; ++cell)
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          // the results which goes to FEEvaluation::submit_gradient()
          DoNotOptimize(
            res = contract<1, 1>(
              grad_Nx_v,
              cached_tensor2(cell, q))); // use the fact that \tau is symmetric
        }
    LIKWID_MARKER_STOP("tensor2_c11");
    timer.leave_subsection();
  }

  // OPTION3: manual
  {
    timer.enter_subsection("tensor2_manual");
    LIKWID_MARKER_START("tensor2_manual");
    for (unsigned int cell = 0; cell < n_cells; ++cell)
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          // the results which goes to FEEvaluation::submit_gradient()
          Tensor<2, dim, VectorizedArray<number>>        res;
          const Tensor<2, dim, VectorizedArray<number>> &rhs =
            cached_tensor2(cell, q);
          for (unsigned int i = 0; i < dim; ++i)
            {
              auto &res_i       = res[i];
              auto &grad_Nx_v_i = grad_Nx_v[i];
              for (unsigned int j = 0; j < dim; ++j)
                {
                  auto &res_ij = res_i[j];
                  auto &rhs_j  = rhs[j];
                  res_ij       = zero;
                  for (unsigned int k = 0; k < dim; ++k)
                    DoNotOptimize(res_ij +=
                                  grad_Nx_v_i[k] *
                                  rhs_j[k]); // rhs is symmetric, actually
                }
            }
        }
    LIKWID_MARKER_STOP("tensor2_manual");
    timer.leave_subsection();
  }

  // OPTION4: manual symmetric
  {
    timer.enter_subsection("tensor2_manual_sym");
    LIKWID_MARKER_START("tensor2_manual_sym");
    for (unsigned int cell = 0; cell < n_cells; ++cell)
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          // the results which goes to FEEvaluation::submit_gradient()
          Tensor<2, dim, VectorizedArray<number>>                 res;
          const SymmetricTensor<2, dim, VectorizedArray<number>> &rhs =
            cached_tensor2sym(cell, q);
          for (unsigned int i = 0; i < dim; ++i)
            {
              auto &res_i       = res[i];
              auto &grad_Nx_v_i = grad_Nx_v[i];
              for (unsigned int j = 0; j < dim; ++j)
                {
                  auto &res_ij = res_i[j];
                  res_ij       = zero;
                  for (unsigned int k = 0; k < dim; ++k)
                    DoNotOptimize(res_ij +=
                                  grad_Nx_v_i[k] *
                                  rhs[j][k]); // rhs is symmetric, actually
                }
            }
        }
    LIKWID_MARKER_STOP("tensor2_manual_sym");
    timer.leave_subsection();
  }

  LIKWID_MARKER_CLOSE;
}


int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  const int dim = (argc > 1 ? Utilities::string_to_int(argv[1]) : 2);

  // note that degree (second parameter) will only affect the inner loop
  // size over quadrature points.
  if (dim == 2)
    {
      test<2, 4>();
    }
  else
    {
      test<3, 4>();
    }

  return 0;
}
