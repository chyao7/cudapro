/*
 * EpilogueVisitor: GEMM 写回 qkv，同时对 q/k 区段 atomic 累加 sum(x²)
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/scale_type.h"

namespace cutlass {
namespace kernel {

template <
    typename ThreadblockShape_, int ThreadCount_,
    typename OutputTileIterator_, typename AccumulatorTile_,
    typename ElementAccumulator_, typename ElementCompute_,
    typename ElementwiseFunctor_>
class EpilogueVisitorQkRmsNorm {
public:
    using ThreadblockShape = ThreadblockShape_;
    static int const kThreadCount = ThreadCount_;
    using OutputTileIterator = OutputTileIterator_;
    using AccumulatorTile = AccumulatorTile_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;
    using ElementwiseFunctor = ElementwiseFunctor_;

    static int const kIterations = OutputTileIterator::kIterations;
    static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;
    static int const kRowIterations = OutputTileIterator::ThreadMap::Iterations::kRow;
    static int const kDeltaRow = OutputTileIterator::ThreadMap::Delta::kRow;

    using ElementOutput = typename OutputTileIterator::Element;
    using LayoutOutput = cutlass::layout::RowMajor;
    using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
    using ComputeFragment = Array<ElementCompute, kElementsPerAccess>;
    using OutputVector = Array<ElementOutput, kElementsPerAccess>;
    using TensorRefD = TensorRef<ElementOutput, LayoutOutput>;

    struct Arguments {
        typename ElementwiseFunctor::Params elementwise;
        TensorRefD ref_C;
        TensorRefD ref_D;
        float *ptr_sum_sq_q{};
        float *ptr_sum_sq_k{};
        int q_size{};
        int kv_size{};
        int head_dim{};
        int num_q_heads{};
        int num_kv_heads{};

        Arguments() = default;

        Arguments(typename ElementwiseFunctor::Params elementwise_,
                  TensorRefD ref_C_, TensorRefD ref_D_, float *ptr_sum_sq_q_,
                  float *ptr_sum_sq_k_, int q_size_, int kv_size_,
                  int head_dim_, int num_q_heads_, int num_kv_heads_)
            : elementwise(elementwise_), ref_C(ref_C_), ref_D(ref_D_),
              ptr_sum_sq_q(ptr_sum_sq_q_), ptr_sum_sq_k(ptr_sum_sq_k_),
              q_size(q_size_), kv_size(kv_size_), head_dim(head_dim_),
              num_q_heads(num_q_heads_), num_kv_heads(num_kv_heads_) {}
    };

    struct Params {
        typename ElementwiseFunctor::Params elementwise;
        typename OutputTileIterator::Params params_C;
        typename OutputTileIterator::Params params_D;
        typename OutputTileIterator::Element *ptr_C{};
        typename OutputTileIterator::Element *ptr_D{};
        float *ptr_sum_sq_q{};
        float *ptr_sum_sq_k{};
        int q_size{};
        int kv_size{};
        int head_dim{};
        int num_q_heads{};
        int num_kv_heads{};

        CUTLASS_HOST_DEVICE
        Params() = default;

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args)
            : elementwise(args.elementwise), params_C(args.ref_C.layout()),
              params_D(args.ref_D.layout()), ptr_C(args.ref_C.data()),
              ptr_D(args.ref_D.data()), ptr_sum_sq_q(args.ptr_sum_sq_q),
              ptr_sum_sq_k(args.ptr_sum_sq_k), q_size(args.q_size),
              kv_size(args.kv_size), head_dim(args.head_dim),
              num_q_heads(args.num_q_heads), num_kv_heads(args.num_kv_heads) {}
    };

    struct SharedStorage {};

private:
    Params const &params_;
    SharedStorage &shared_storage_;
    MatrixCoord extent_;
    ElementwiseFunctor elementwise_;
    OutputTileIterator iterator_C_;
    OutputTileIterator iterator_D_;
    typename OutputTileIterator::Fragment fragment_C_;
    typename OutputTileIterator::Fragment fragment_D_;
    ElementAccumulator alpha_;
    ElementAccumulator beta_;
    MatrixCoord thread_offset_;

public:
    CUTLASS_DEVICE
    EpilogueVisitorQkRmsNorm(
        Params const &params, SharedStorage &shared_storage,
        MatrixCoord const &problem_size, int thread_idx, int, int,
        MatrixCoord const &threadblock_offset = MatrixCoord(0, 0))
        : params_(params), shared_storage_(shared_storage), extent_(problem_size),
          elementwise_(params.elementwise),
          iterator_C_(params.params_C, params.ptr_C, problem_size, thread_idx,
                      threadblock_offset),
          iterator_D_(params.params_D, params.ptr_D, problem_size, thread_idx,
                      threadblock_offset) {
        alpha_ = (params.elementwise.alpha_ptr ? *params.elementwise.alpha_ptr
                                               : params.elementwise.alpha);
        beta_ = (params.elementwise.beta_ptr ? *params.elementwise.beta_ptr
                                             : params.elementwise.beta);
        if (beta_ == ElementAccumulator()) {
            iterator_C_.clear_mask();
        }
    }

    CUTLASS_DEVICE
    void set_k_partition(int, int) {}

    CUTLASS_DEVICE
    void set_batch_index(int) {}

    CUTLASS_DEVICE
    void begin_epilogue() {}

    CUTLASS_DEVICE
    void begin_step(int) {
        fragment_D_.clear();
        if (elementwise_.kScale !=
            cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling) {
            fragment_C_.clear();
            iterator_C_.load(fragment_C_);
            ++iterator_C_;
        }
    }

    CUTLASS_DEVICE
    void begin_row(int) {}

    CUTLASS_DEVICE
    void visit(int, int, int, int frag_idx, AccumulatorFragment const &accum) {
        thread_offset_ =
            iterator_D_.thread_start() +
            OutputTileIterator::ThreadMap::iteration_offset(frag_idx);

        NumericArrayConverter<ElementCompute, ElementOutput, kElementsPerAccess>
            source_converter;
        OutputVector &source_vector =
            reinterpret_cast<OutputVector *>(&fragment_C_)[frag_idx];

        ComputeFragment result;
        if (elementwise_.kScale ==
            cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling) {
            result = source_converter(elementwise_(accum));
        } else {
            result = source_converter(elementwise_(accum, source_vector));
        }

        int const global_row = thread_offset_.row();
        int const base_col = thread_offset_.column();
        bool const column_guard = (thread_offset_.column() < extent_.column());

        if (column_guard) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kElementsPerAccess; ++i) {
                int const col = base_col + i;
                if (global_row >= extent_.row() || col >= extent_.column()) {
                    continue;
                }
                float const val = static_cast<float>(result[i]);
                if (col < params_.q_size) {
                    int const head = col / params_.head_dim;
                    int const idx = global_row * params_.num_q_heads + head;
                    atomicAdd(params_.ptr_sum_sq_q + idx, val * val);
                } else if (col < params_.q_size + params_.kv_size) {
                    int const col_k = col - params_.q_size;
                    int const head = col_k / params_.head_dim;
                    int const idx = global_row * params_.num_kv_heads + head;
                    atomicAdd(params_.ptr_sum_sq_k + idx, val * val);
                }
            }
        }

        NumericArrayConverter<ElementOutput, ElementCompute, kElementsPerAccess>
            output_converter;
        OutputVector &output =
            reinterpret_cast<OutputVector *>(&fragment_D_)[frag_idx];
        output = output_converter(result);
    }

    CUTLASS_DEVICE
    void end_row(int) {}

    CUTLASS_DEVICE
    void end_step(int) {
        iterator_D_.store(fragment_D_);
        ++iterator_D_;
    }

    CUTLASS_DEVICE
    void end_epilogue() {}
};

} // namespace kernel
} // namespace cutlass
