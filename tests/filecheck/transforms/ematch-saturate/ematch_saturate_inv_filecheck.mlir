// CHECK:func.func @get_inv(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %0 = equivalence.graph : () -> tensor<4x4xf32> {
// CHECK-NEXT:      %c0 = equivalence.class %c0_1 : index
// CHECK-NEXT:      %c0_1 = arith.constant 0 : index
// CHECK-NEXT:      %c1 = equivalence.class %c1_1 : index
// CHECK-NEXT:      %c1_1 = arith.constant 1 : index
// CHECK-NEXT:      %c4 = equivalence.class %c4_1 : index
// CHECK-NEXT:      %c4_1 = arith.constant 4 : index
// CHECK-NEXT:      %cst = equivalence.class %cst_1 : f32
// CHECK-NEXT:      %cst_1 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:      %cst_2 = equivalence.class %cst_3 : f32
// CHECK-NEXT:      %cst_3 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %alloc = equivalence.class %alloc_1 : tensor<4x4xf32>
// CHECK-NEXT:      %alloc_1 = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:      %init_alloc = equivalence.class %init_alloc_1 : tensor<4x4xf32>
// CHECK-NEXT:      %init_alloc_1 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%t_alloc = %alloc) -> (tensor<4x4xf32>) {
// CHECK-NEXT:        %new_alloc = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%t_alloc_inner = %t_alloc) -> (tensor<4x4xf32>) {
// CHECK-NEXT:          %1 = arith.cmpi eq, %arg1, %arg2 : index
// CHECK-NEXT:          %2 = arith.select %1, %cst, %cst_2 : f32
// CHECK-NEXT:          %3 = tensor.insert %2 into %t_alloc_inner[%arg1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:          scf.yield %3 : tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %new_alloc : tensor<4x4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %final_alloc = equivalence.class %final_alloc_1 : tensor<4x4xf32>
// CHECK-NEXT:      %final_alloc_2, %final_alloc_1 = scf.for %arg1_1 = %c0 to %c4 step %c1 iter_args(%outer_alloc = %init_alloc, %outer_alloc_1 = %arg0) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:        %4 = tensor.extract %outer_alloc_1[%arg1_1, %arg1_1] : tensor<4x4xf32>
// CHECK-NEXT:        %norm_alloc, %norm_alloc_1 = scf.for %arg2_1 = %c0 to %c4 step %c1 iter_args(%norm_a = %outer_alloc, %norm_a1 = %outer_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %5 = tensor.extract %norm_a1[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          %6 = arith.divf %5, %4 : f32
// CHECK-NEXT:          %7 = tensor.insert %6 into %norm_a1[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          %8 = tensor.extract %norm_a[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          %9 = arith.divf %8, %4 : f32
// CHECK-NEXT:          %10 = tensor.insert %9 into %norm_a[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          scf.yield %10, %7 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %elim_alloc, %elim_alloc_1 = scf.for %arg2_2 = %c0 to %c4 step %c1 iter_args(%elim_a = %norm_alloc, %elim_a1 = %norm_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %11 = arith.cmpi ne, %arg2_2, %arg1_1 : index
// CHECK-NEXT:          %res_a, %res_a1 = scf.if %11 -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:            %12 = tensor.extract %elim_a1[%arg2_2, %arg1_1] : tensor<4x4xf32>
// CHECK-NEXT:            %sub_alloc, %sub_alloc_1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%sub_a = %elim_a, %sub_a1 = %elim_a1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:              %13 = tensor.extract %sub_a1[%arg1_1, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %14 = tensor.extract %sub_a1[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %15 = arith.mulf %12, %13 : f32
// CHECK-NEXT:              %16 = arith.subf %14, %15 : f32
// CHECK-NEXT:              %17 = tensor.insert %16 into %sub_a1[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %18 = tensor.extract %sub_a[%arg1_1, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %19 = tensor.extract %sub_a[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %20 = arith.mulf %12, %18 : f32
// CHECK-NEXT:              %21 = arith.subf %19, %20 : f32
// CHECK-NEXT:              %22 = tensor.insert %21 into %sub_a[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              scf.yield %22, %17 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %sub_alloc, %sub_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:          } else {
// CHECK-NEXT:            scf.yield %elim_a, %elim_a1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield %res_a, %res_a1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %elim_alloc, %elim_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %final_alloc_3 = equivalence.class %23, %final_alloc_2 : tensor<4x4xf32>
// CHECK-NEXT:      %23 = func.call @matrix_inverse_4x4(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:      equivalence.yield %final_alloc_3 : tensor<4x4xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %0 : tensor<4x4xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @matrix_inverse_4x4(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %0 = equivalence.graph : () -> tensor<4x4xf32> {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %c4 = arith.constant 4 : index
// CHECK-NEXT:      %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:      %cst_1 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %alloc = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:      %init_alloc = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%t_alloc = %alloc) -> (tensor<4x4xf32>) {
// CHECK-NEXT:        %new_alloc = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%t_alloc_inner = %t_alloc) -> (tensor<4x4xf32>) {
// CHECK-NEXT:          %1 = arith.cmpi eq, %arg1, %arg2 : index
// CHECK-NEXT:          %2 = arith.select %1, %cst, %cst_1 : f32
// CHECK-NEXT:          %3 = tensor.insert %2 into %t_alloc_inner[%arg1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:          scf.yield %3 : tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %new_alloc : tensor<4x4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %final_alloc, %final_alloc_1 = scf.for %arg1_1 = %c0 to %c4 step %c1 iter_args(%outer_alloc = %init_alloc, %outer_alloc_1 = %arg0) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:        %4 = tensor.extract %outer_alloc_1[%arg1_1, %arg1_1] : tensor<4x4xf32>
// CHECK-NEXT:        %norm_alloc, %norm_alloc_1 = scf.for %arg2_1 = %c0 to %c4 step %c1 iter_args(%norm_a = %outer_alloc, %norm_a1 = %outer_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %5 = tensor.extract %norm_a1[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          %6 = arith.divf %5, %4 : f32
// CHECK-NEXT:          %7 = tensor.insert %6 into %norm_a1[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          %8 = tensor.extract %norm_a[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          %9 = arith.divf %8, %4 : f32
// CHECK-NEXT:          %10 = tensor.insert %9 into %norm_a[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          scf.yield %10, %7 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %elim_alloc, %elim_alloc_1 = scf.for %arg2_2 = %c0 to %c4 step %c1 iter_args(%elim_a = %norm_alloc, %elim_a1 = %norm_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %11 = arith.cmpi ne, %arg2_2, %arg1_1 : index
// CHECK-NEXT:          %res_a, %res_a1 = scf.if %11 -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:            %12 = tensor.extract %elim_a1[%arg2_2, %arg1_1] : tensor<4x4xf32>
// CHECK-NEXT:            %sub_alloc, %sub_alloc_1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%sub_a = %elim_a, %sub_a1 = %elim_a1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:              %13 = tensor.extract %sub_a1[%arg1_1, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %14 = tensor.extract %sub_a1[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %15 = arith.mulf %12, %13 : f32
// CHECK-NEXT:              %16 = arith.subf %14, %15 : f32
// CHECK-NEXT:              %17 = tensor.insert %16 into %sub_a1[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %18 = tensor.extract %sub_a[%arg1_1, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %19 = tensor.extract %sub_a[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %20 = arith.mulf %12, %18 : f32
// CHECK-NEXT:              %21 = arith.subf %19, %20 : f32
// CHECK-NEXT:              %22 = tensor.insert %21 into %sub_a[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              scf.yield %22, %17 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %sub_alloc, %sub_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:          } else {
// CHECK-NEXT:            scf.yield %elim_a, %elim_a1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield %res_a, %res_a1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %elim_alloc, %elim_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      equivalence.yield %final_alloc : tensor<4x4xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %0 : tensor<4x4xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func private @dot(tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:  func.func private @solve(tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:  func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %0 = equivalence.graph : () -> tensor<4x4xf32> {
// CHECK-NEXT:      %c0 = equivalence.class %c0_1 : index
// CHECK-NEXT:      %c0_1 = arith.constant 0 : index
// CHECK-NEXT:      %c1 = equivalence.class %c1_1 : index
// CHECK-NEXT:      %c1_1 = arith.constant 1 : index
// CHECK-NEXT:      %c4 = equivalence.class %c4_1 : index
// CHECK-NEXT:      %c4_1 = arith.constant 4 : index
// CHECK-NEXT:      %cst = equivalence.class %cst_1 : f32
// CHECK-NEXT:      %cst_1 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:      %cst_2 = equivalence.class %cst_3 : f32
// CHECK-NEXT:      %cst_3 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %alloc = equivalence.class %alloc_1 : tensor<4x4xf32>
// CHECK-NEXT:      %alloc_1 = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:      %init_alloc = equivalence.class %init_alloc_1 : tensor<4x4xf32>
// CHECK-NEXT:      %init_alloc_1 = scf.for %arg1_1 = %c0 to %c4 step %c1 iter_args(%t_alloc = %alloc) -> (tensor<4x4xf32>) {
// CHECK-NEXT:        %new_alloc = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%t_alloc_inner = %t_alloc) -> (tensor<4x4xf32>) {
// CHECK-NEXT:          %1 = arith.cmpi eq, %arg1_1, %arg2 : index
// CHECK-NEXT:          %2 = arith.select %1, %cst, %cst_2 : f32
// CHECK-NEXT:          %3 = tensor.insert %2 into %t_alloc_inner[%arg1_1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:          scf.yield %3 : tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %new_alloc : tensor<4x4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %final_alloc = equivalence.class %final_alloc_1 : tensor<4x4xf32>
// CHECK-NEXT:      %final_alloc_2, %final_alloc_1 = scf.for %arg1_2 = %c0 to %c4 step %c1 iter_args(%outer_alloc = %init_alloc, %outer_alloc_1 = %arg0) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:        %4 = tensor.extract %outer_alloc_1[%arg1_2, %arg1_2] : tensor<4x4xf32>
// CHECK-NEXT:        %norm_alloc, %norm_alloc_1 = scf.for %arg2_1 = %c0 to %c4 step %c1 iter_args(%norm_a = %outer_alloc, %norm_a1 = %outer_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %5 = tensor.extract %norm_a1[%arg1_2, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          %6 = arith.divf %5, %4 : f32
// CHECK-NEXT:          %7 = tensor.insert %6 into %norm_a1[%arg1_2, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          %8 = tensor.extract %norm_a[%arg1_2, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          %9 = arith.divf %8, %4 : f32
// CHECK-NEXT:          %10 = tensor.insert %9 into %norm_a[%arg1_2, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          scf.yield %10, %7 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %elim_alloc, %elim_alloc_1 = scf.for %arg2_2 = %c0 to %c4 step %c1 iter_args(%elim_a = %norm_alloc, %elim_a1 = %norm_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %11 = arith.cmpi ne, %arg2_2, %arg1_2 : index
// CHECK-NEXT:          %res_a, %res_a1 = scf.if %11 -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:            %12 = tensor.extract %elim_a1[%arg2_2, %arg1_2] : tensor<4x4xf32>
// CHECK-NEXT:            %sub_alloc, %sub_alloc_1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%sub_a = %elim_a, %sub_a1 = %elim_a1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:              %13 = tensor.extract %sub_a1[%arg1_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %14 = tensor.extract %sub_a1[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %15 = arith.mulf %12, %13 : f32
// CHECK-NEXT:              %16 = arith.subf %14, %15 : f32
// CHECK-NEXT:              %17 = tensor.insert %16 into %sub_a1[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %18 = tensor.extract %sub_a[%arg1_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %19 = tensor.extract %sub_a[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %20 = arith.mulf %12, %18 : f32
// CHECK-NEXT:              %21 = arith.subf %19, %20 : f32
// CHECK-NEXT:              %22 = tensor.insert %21 into %sub_a[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              scf.yield %22, %17 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %sub_alloc, %sub_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:          } else {
// CHECK-NEXT:            scf.yield %elim_a, %elim_a1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield %res_a, %res_a1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %elim_alloc, %elim_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %23 = func.call @matrix_inverse_4x4(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:      %final_alloc_3 = equivalence.class %24, %23, %final_alloc_2 : tensor<4x4xf32>
// CHECK-NEXT:      %24 = func.call @get_inv(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:      %25 = func.call @solve(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:      %26 = equivalence.class %27, %25 : tensor<4x4xf32>
// CHECK-NEXT:      %27 = func.call @dot(%final_alloc_3, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:      equivalence.yield %26 : tensor<4x4xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %0 : tensor<4x4xf32>
// CHECK-NEXT:  }