// CHECK: func.func @get_inv(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:     %0 = equivalence.graph : () -> tensor<4x4xf32> {
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       %c4 = arith.constant 4 : index
// CHECK-NEXT:       %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:       %cst_1 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:       %alloc = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:       %alloc_1 = equivalence.class %alloc : tensor<4x4xf32>
// CHECK-NEXT:       %init_alloc, %init_alloc_1 = scf.for %arg1 = %c0_1 to %c4_1 step %c1_1 iter_args(%t_alloc = %alloc_1, %t_alloc_1 = %alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:         %new_alloc, %new_alloc_1 = scf.for %arg2 = %c0_1 to %c4_1 step %c1_1 iter_args(%t_alloc_inner = %t_alloc, %t_alloc_1_inner = %t_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:           %1 = arith.cmpi eq, %arg1, %arg2 : index
// CHECK-NEXT:           %2 = arith.select %1, %cst_2, %cst_3 : f32
// CHECK-NEXT:           %3 = tensor.insert %2 into %t_alloc_inner[%arg1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:           %4 = tensor.extract %arg0[%arg1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:           %5 = tensor.insert %4 into %t_alloc_1_inner[%arg1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:           scf.yield %3, %5 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %new_alloc, %new_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       %final_alloc, %final_alloc_1 = scf.for %arg1_1 = %c0_1 to %c4_1 step %c1_1 iter_args(%outer_alloc = %init_alloc_2, %outer_alloc_1 = %init_alloc_3) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:         %6 = tensor.extract %outer_alloc_1[%arg1_1, %arg1_1] : tensor<4x4xf32>
// CHECK-NEXT:         %norm_alloc, %norm_alloc_1 = scf.for %arg2_1 = %c0_1 to %c4_1 step %c1_1 iter_args(%norm_a = %outer_alloc, %norm_a1 = %outer_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:           %7 = tensor.extract %norm_a1[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:           %8 = arith.divf %7, %6 : f32
// CHECK-NEXT:           %9 = tensor.insert %8 into %norm_a1[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:           %10 = tensor.extract %norm_a[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:           %11 = arith.divf %10, %6 : f32
// CHECK-NEXT:           %12 = tensor.insert %11 into %norm_a[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:           scf.yield %12, %9 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         %elim_alloc, %elim_alloc_1 = scf.for %arg2_2 = %c0_1 to %c4_1 step %c1_1 iter_args(%elim_a = %norm_alloc, %elim_a1 = %norm_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:           %13 = arith.cmpi ne, %arg2_2, %arg1_1 : index
// CHECK-NEXT:           %res_a, %res_a1 = scf.if %13 -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:             %14 = tensor.extract %elim_a1[%arg2_2, %arg1_1] : tensor<4x4xf32>
// CHECK-NEXT:             %sub_alloc, %sub_alloc_1 = scf.for %arg3 = %c0_1 to %c4_1 step %c1_1 iter_args(%sub_a = %elim_a, %sub_a1 = %elim_a1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:               %15 = tensor.extract %sub_a1[%arg1_1, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:               %16 = tensor.extract %sub_a1[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:               %17 = arith.mulf %14, %15 : f32
// CHECK-NEXT:               %18 = arith.subf %16, %17 : f32
// CHECK-NEXT:               %19 = tensor.insert %18 into %sub_a1[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:               %20 = tensor.extract %sub_a[%arg1_1, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:               %21 = tensor.extract %sub_a[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:               %22 = arith.mulf %14, %20 : f32
// CHECK-NEXT:               %23 = arith.subf %21, %22 : f32
// CHECK-NEXT:               %24 = tensor.insert %23 into %sub_a[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:               scf.yield %24, %19 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:             scf.yield %sub_alloc, %sub_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:           } else {
// CHECK-NEXT:             scf.yield %elim_a, %elim_a1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %res_a, %res_a1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %elim_alloc, %elim_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       %25 = equivalence.class %26, %final_alloc, %27 : tensor<4x4xf32>
// CHECK-NEXT:       %c0_1 = equivalence.class %c0 : index
// CHECK-NEXT:       %c1_1 = equivalence.class %c1 : index
// CHECK-NEXT:       %c4_1 = equivalence.class %c4 : index
// CHECK-NEXT:       %cst_2 = equivalence.class %cst : f32
// CHECK-NEXT:       %cst_3 = equivalence.class %cst_1 : f32
// CHECK-NEXT:       %init_alloc_2 = equivalence.class %init_alloc : tensor<4x4xf32>
// CHECK-NEXT:       %init_alloc_3 = equivalence.class %init_alloc_1 : tensor<4x4xf32>
// CHECK-NEXT:       %final_alloc_2 = equivalence.class %final_alloc_1 : tensor<4x4xf32>
// CHECK-NEXT:       %27 = scf.execute_region -> (tensor<4x4xf32>) {
// CHECK-NEXT:         %c0_2 = arith.constant 0 : index
// CHECK-NEXT:         %c1_2 = arith.constant 1 : index
// CHECK-NEXT:         %c4_2 = arith.constant 4 : index
// CHECK-NEXT:         %cst_4 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:         %cst_5 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:         %alloc_2 = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:         %alloc_3 = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:         %init_alloc_4, %init_alloc_5 = scf.for %arg1_2 = %c0_2 to %c4_2 step %c1_2 iter_args(%t_alloc_2 = %alloc_2, %t_alloc_3 = %alloc_3) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:           %new_alloc_2, %new_alloc_3 = scf.for %arg2_3 = %c0_2 to %c4_2 step %c1_2 iter_args(%t_alloc_inner_1 = %t_alloc_2, %t_alloc_1_inner_1 = %t_alloc_3) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:             %28 = arith.cmpi eq, %arg1_2, %arg2_3 : index
// CHECK-NEXT:             %29 = arith.select %28, %cst_4, %cst_5 : f32
// CHECK-NEXT:             %30 = tensor.insert %29 into %t_alloc_inner_1[%arg1_2, %arg2_3] : tensor<4x4xf32>
// CHECK-NEXT:             %31 = tensor.extract %arg0[%arg1_2, %arg2_3] : tensor<4x4xf32>
// CHECK-NEXT:             %32 = tensor.insert %31 into %t_alloc_1_inner_1[%arg1_2, %arg2_3] : tensor<4x4xf32>
// CHECK-NEXT:             scf.yield %30, %32 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %new_alloc_2, %new_alloc_3 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         %final_alloc_3, %final_alloc_4 = scf.for %arg1_3 = %c0_2 to %c4_2 step %c1_2 iter_args(%outer_alloc_2 = %init_alloc_4, %outer_alloc_3 = %init_alloc_5) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:           %33 = tensor.extract %outer_alloc_3[%arg1_3, %arg1_3] : tensor<4x4xf32>
// CHECK-NEXT:           %norm_alloc_2, %norm_alloc_3 = scf.for %arg2_4 = %c0_2 to %c4_2 step %c1_2 iter_args(%norm_a_1 = %outer_alloc_2, %norm_a1_1 = %outer_alloc_3) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:             %34 = tensor.extract %norm_a1_1[%arg1_3, %arg2_4] : tensor<4x4xf32>
// CHECK-NEXT:             %35 = arith.divf %34, %33 : f32
// CHECK-NEXT:             %36 = tensor.insert %35 into %norm_a1_1[%arg1_3, %arg2_4] : tensor<4x4xf32>
// CHECK-NEXT:             %37 = tensor.extract %norm_a_1[%arg1_3, %arg2_4] : tensor<4x4xf32>
// CHECK-NEXT:             %38 = arith.divf %37, %33 : f32
// CHECK-NEXT:             %39 = tensor.insert %38 into %norm_a_1[%arg1_3, %arg2_4] : tensor<4x4xf32>
// CHECK-NEXT:             scf.yield %39, %36 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           %elim_alloc_2, %elim_alloc_3 = scf.for %arg2_5 = %c0_2 to %c4_2 step %c1_2 iter_args(%elim_a_1 = %norm_alloc_2, %elim_a1_1 = %norm_alloc_3) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:             %40 = arith.cmpi ne, %arg2_5, %arg1_3 : index
// CHECK-NEXT:             %res_a_1, %res_a1_1 = scf.if %40 -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:               %41 = tensor.extract %elim_a1_1[%arg2_5, %arg1_3] : tensor<4x4xf32>
// CHECK-NEXT:               %sub_alloc_2, %sub_alloc_3 = scf.for %arg3_1 = %c0_2 to %c4_2 step %c1_2 iter_args(%sub_a_1 = %elim_a_1, %sub_a1_1 = %elim_a1_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:                 %42 = tensor.extract %sub_a1_1[%arg1_3, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:                 %43 = tensor.extract %sub_a1_1[%arg2_5, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:                 %44 = arith.mulf %41, %42 : f32
// CHECK-NEXT:                 %45 = arith.subf %43, %44 : f32
// CHECK-NEXT:                 %46 = tensor.insert %45 into %sub_a1_1[%arg2_5, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:                 %47 = tensor.extract %sub_a_1[%arg1_3, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:                 %48 = tensor.extract %sub_a_1[%arg2_5, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:                 %49 = arith.mulf %41, %47 : f32
// CHECK-NEXT:                 %50 = arith.subf %48, %49 : f32
// CHECK-NEXT:                 %51 = tensor.insert %50 into %sub_a_1[%arg2_5, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:                 scf.yield %51, %46 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:               }
// CHECK-NEXT:               scf.yield %sub_alloc_2, %sub_alloc_3 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:             } else {
// CHECK-NEXT:               scf.yield %elim_a_1, %elim_a1_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:             scf.yield %res_a_1, %res_a1_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %elim_alloc_2, %elim_alloc_3 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %final_alloc_3 : tensor<4x4xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       %26 = func.call @matrix_inverse_4x4(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:       equivalence.yield %25 : tensor<4x4xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return %0 : tensor<4x4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @matrix_inverse_4x4(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:     %0 = equivalence.graph : () -> tensor<4x4xf32> {
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       %c4 = arith.constant 4 : index
// CHECK-NEXT:       %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:       %cst_1 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:       %alloc = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:       %alloc_1 = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:       %init_alloc, %init_alloc_1 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%t_alloc = %alloc, %t_alloc_1 = %alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:         %new_alloc, %new_alloc_1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%t_alloc_inner = %t_alloc, %t_alloc_1_inner = %t_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:           %1 = arith.cmpi eq, %arg1, %arg2 : index
// CHECK-NEXT:           %2 = arith.select %1, %cst, %cst_1 : f32
// CHECK-NEXT:           %3 = tensor.insert %2 into %t_alloc_inner[%arg1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:           %4 = tensor.extract %arg0[%arg1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:           %5 = tensor.insert %4 into %t_alloc_1_inner[%arg1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:           scf.yield %3, %5 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %new_alloc, %new_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       %final_alloc, %final_alloc_1 = scf.for %arg1_1 = %c0 to %c4 step %c1 iter_args(%outer_alloc = %init_alloc, %outer_alloc_1 = %init_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:         %6 = tensor.extract %outer_alloc_1[%arg1_1, %arg1_1] : tensor<4x4xf32>
// CHECK-NEXT:         %norm_alloc, %norm_alloc_1 = scf.for %arg2_1 = %c0 to %c4 step %c1 iter_args(%norm_a = %outer_alloc, %norm_a1 = %outer_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:           %7 = tensor.extract %norm_a1[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:           %8 = arith.divf %7, %6 : f32
// CHECK-NEXT:           %9 = tensor.insert %8 into %norm_a1[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:           %10 = tensor.extract %norm_a[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:           %11 = arith.divf %10, %6 : f32
// CHECK-NEXT:           %12 = tensor.insert %11 into %norm_a[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:           scf.yield %12, %9 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         %elim_alloc, %elim_alloc_1 = scf.for %arg2_2 = %c0 to %c4 step %c1 iter_args(%elim_a = %norm_alloc, %elim_a1 = %norm_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:           %13 = arith.cmpi ne, %arg2_2, %arg1_1 : index
// CHECK-NEXT:           %res_a, %res_a1 = scf.if %13 -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:             %14 = tensor.extract %elim_a1[%arg2_2, %arg1_1] : tensor<4x4xf32>
// CHECK-NEXT:             %sub_alloc, %sub_alloc_1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%sub_a = %elim_a, %sub_a1 = %elim_a1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:               %15 = tensor.extract %sub_a1[%arg1_1, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:               %16 = tensor.extract %sub_a1[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:               %17 = arith.mulf %14, %15 : f32
// CHECK-NEXT:               %18 = arith.subf %16, %17 : f32
// CHECK-NEXT:               %19 = tensor.insert %18 into %sub_a1[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:               %20 = tensor.extract %sub_a[%arg1_1, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:               %21 = tensor.extract %sub_a[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:               %22 = arith.mulf %14, %20 : f32
// CHECK-NEXT:               %23 = arith.subf %21, %22 : f32
// CHECK-NEXT:               %24 = tensor.insert %23 into %sub_a[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:               scf.yield %24, %19 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:             scf.yield %sub_alloc, %sub_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:           } else {
// CHECK-NEXT:             scf.yield %elim_a, %elim_a1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %res_a, %res_a1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %elim_alloc, %elim_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       equivalence.yield %final_alloc : tensor<4x4xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return %0 : tensor<4x4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @dot(tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:   func.func private @solve(tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:   func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:     %0 = equivalence.graph : () -> tensor<4x4xf32> {
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       %c4 = arith.constant 4 : index
// CHECK-NEXT:       %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:       %cst_1 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:       %alloc = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:       %init_alloc, %init_alloc_1 = scf.for %arg1_1 = %c0_1 to %c4_1 step %c1_1 iter_args(%t_alloc = %alloc_1, %t_alloc_1 = %alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:         %new_alloc, %new_alloc_1 = scf.for %arg2 = %c0_1 to %c4_1 step %c1_1 iter_args(%t_alloc_inner = %t_alloc, %t_alloc_1_inner = %t_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:           %1 = arith.cmpi eq, %arg1_1, %arg2 : index
// CHECK-NEXT:           %2 = arith.select %1, %cst_2, %cst_3 : f32
// CHECK-NEXT:           %3 = tensor.insert %2 into %t_alloc_inner[%arg1_1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:           %4 = tensor.extract %arg0[%arg1_1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:           %5 = tensor.insert %4 into %t_alloc_1_inner[%arg1_1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:           scf.yield %3, %5 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %new_alloc, %new_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       %final_alloc, %final_alloc_1 = scf.for %arg1_2 = %c0_1 to %c4_1 step %c1_1 iter_args(%outer_alloc = %init_alloc_2, %outer_alloc_1 = %init_alloc_3) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:         %6 = tensor.extract %outer_alloc_1[%arg1_2, %arg1_2] : tensor<4x4xf32>
// CHECK-NEXT:         %norm_alloc, %norm_alloc_1 = scf.for %arg2_1 = %c0_1 to %c4_1 step %c1_1 iter_args(%norm_a = %outer_alloc, %norm_a1 = %outer_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:           %7 = tensor.extract %norm_a1[%arg1_2, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:           %8 = arith.divf %7, %6 : f32
// CHECK-NEXT:           %9 = tensor.insert %8 into %norm_a1[%arg1_2, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:           %10 = tensor.extract %norm_a[%arg1_2, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:           %11 = arith.divf %10, %6 : f32
// CHECK-NEXT:           %12 = tensor.insert %11 into %norm_a[%arg1_2, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:           scf.yield %12, %9 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         %elim_alloc, %elim_alloc_1 = scf.for %arg2_2 = %c0_1 to %c4_1 step %c1_1 iter_args(%elim_a = %norm_alloc, %elim_a1 = %norm_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:           %13 = arith.cmpi ne, %arg2_2, %arg1_2 : index
// CHECK-NEXT:           %res_a, %res_a1 = scf.if %13 -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:             %14 = tensor.extract %elim_a1[%arg2_2, %arg1_2] : tensor<4x4xf32>
// CHECK-NEXT:             %sub_alloc, %sub_alloc_1 = scf.for %arg3 = %c0_1 to %c4_1 step %c1_1 iter_args(%sub_a = %elim_a, %sub_a1 = %elim_a1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:               %15 = tensor.extract %sub_a1[%arg1_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:               %16 = tensor.extract %sub_a1[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:               %17 = arith.mulf %14, %15 : f32
// CHECK-NEXT:               %18 = arith.subf %16, %17 : f32
// CHECK-NEXT:               %19 = tensor.insert %18 into %sub_a1[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:               %20 = tensor.extract %sub_a[%arg1_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:               %21 = tensor.extract %sub_a[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:               %22 = arith.mulf %14, %20 : f32
// CHECK-NEXT:               %23 = arith.subf %21, %22 : f32
// CHECK-NEXT:               %24 = tensor.insert %23 into %sub_a[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:               scf.yield %24, %19 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:             scf.yield %sub_alloc, %sub_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:           } else {
// CHECK-NEXT:             scf.yield %elim_a, %elim_a1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %res_a, %res_a1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %elim_alloc, %elim_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       %c0_1 = equivalence.class %c0 : index
// CHECK-NEXT:       %c1_1 = equivalence.class %c1 : index
// CHECK-NEXT:       %c4_1 = equivalence.class %c4 : index
// CHECK-NEXT:       %cst_2 = equivalence.class %cst : f32
// CHECK-NEXT:       %cst_3 = equivalence.class %cst_1 : f32
// CHECK-NEXT:       %alloc_1 = equivalence.class %alloc : tensor<4x4xf32>
// CHECK-NEXT:       %init_alloc_2 = equivalence.class %init_alloc : tensor<4x4xf32>
// CHECK-NEXT:       %init_alloc_3 = equivalence.class %init_alloc_1 : tensor<4x4xf32>
// CHECK-NEXT:       %final_alloc_2 = equivalence.class %final_alloc_1 : tensor<4x4xf32>
// CHECK-NEXT:       %25 = scf.execute_region -> (tensor<4x4xf32>) {
// CHECK-NEXT:         %c0_2 = arith.constant 0 : index
// CHECK-NEXT:         %c1_2 = arith.constant 1 : index
// CHECK-NEXT:         %c4_2 = arith.constant 4 : index
// CHECK-NEXT:         %cst_4 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:         %cst_5 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:         %alloc_2 = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:         %alloc_3 = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:         %init_alloc_4, %init_alloc_5 = scf.for %arg1_3 = %c0_2 to %c4_2 step %c1_2 iter_args(%t_alloc_2 = %alloc_2, %t_alloc_3 = %alloc_3) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:           %new_alloc_2, %new_alloc_3 = scf.for %arg2_3 = %c0_2 to %c4_2 step %c1_2 iter_args(%t_alloc_inner_1 = %t_alloc_2, %t_alloc_1_inner_1 = %t_alloc_3) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:             %26 = arith.cmpi eq, %arg1_3, %arg2_3 : index
// CHECK-NEXT:             %27 = arith.select %26, %cst_4, %cst_5 : f32
// CHECK-NEXT:             %28 = tensor.insert %27 into %t_alloc_inner_1[%arg1_3, %arg2_3] : tensor<4x4xf32>
// CHECK-NEXT:             %29 = tensor.extract %arg0[%arg1_3, %arg2_3] : tensor<4x4xf32>
// CHECK-NEXT:             %30 = tensor.insert %29 into %t_alloc_1_inner_1[%arg1_3, %arg2_3] : tensor<4x4xf32>
// CHECK-NEXT:             scf.yield %28, %30 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %new_alloc_2, %new_alloc_3 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         %final_alloc_3, %final_alloc_4 = scf.for %arg1_4 = %c0_2 to %c4_2 step %c1_2 iter_args(%outer_alloc_2 = %init_alloc_4, %outer_alloc_3 = %init_alloc_5) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:           %31 = tensor.extract %outer_alloc_3[%arg1_4, %arg1_4] : tensor<4x4xf32>
// CHECK-NEXT:           %norm_alloc_2, %norm_alloc_3 = scf.for %arg2_4 = %c0_2 to %c4_2 step %c1_2 iter_args(%norm_a_1 = %outer_alloc_2, %norm_a1_1 = %outer_alloc_3) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:             %32 = tensor.extract %norm_a1_1[%arg1_4, %arg2_4] : tensor<4x4xf32>
// CHECK-NEXT:             %33 = arith.divf %32, %31 : f32
// CHECK-NEXT:             %34 = tensor.insert %33 into %norm_a1_1[%arg1_4, %arg2_4] : tensor<4x4xf32>
// CHECK-NEXT:             %35 = tensor.extract %norm_a_1[%arg1_4, %arg2_4] : tensor<4x4xf32>
// CHECK-NEXT:             %36 = arith.divf %35, %31 : f32
// CHECK-NEXT:             %37 = tensor.insert %36 into %norm_a_1[%arg1_4, %arg2_4] : tensor<4x4xf32>
// CHECK-NEXT:             scf.yield %37, %34 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           %elim_alloc_2, %elim_alloc_3 = scf.for %arg2_5 = %c0_2 to %c4_2 step %c1_2 iter_args(%elim_a_1 = %norm_alloc_2, %elim_a1_1 = %norm_alloc_3) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:             %38 = arith.cmpi ne, %arg2_5, %arg1_4 : index
// CHECK-NEXT:             %res_a_1, %res_a1_1 = scf.if %38 -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:               %39 = tensor.extract %elim_a1_1[%arg2_5, %arg1_4] : tensor<4x4xf32>
// CHECK-NEXT:               %sub_alloc_2, %sub_alloc_3 = scf.for %arg3_1 = %c0_2 to %c4_2 step %c1_2 iter_args(%sub_a_1 = %elim_a_1, %sub_a1_1 = %elim_a1_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:                 %40 = tensor.extract %sub_a1_1[%arg1_4, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:                 %41 = tensor.extract %sub_a1_1[%arg2_5, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:                 %42 = arith.mulf %39, %40 : f32
// CHECK-NEXT:                 %43 = arith.subf %41, %42 : f32
// CHECK-NEXT:                 %44 = tensor.insert %43 into %sub_a1_1[%arg2_5, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:                 %45 = tensor.extract %sub_a_1[%arg1_4, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:                 %46 = tensor.extract %sub_a_1[%arg2_5, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:                 %47 = arith.mulf %39, %45 : f32
// CHECK-NEXT:                 %48 = arith.subf %46, %47 : f32
// CHECK-NEXT:                 %49 = tensor.insert %48 into %sub_a_1[%arg2_5, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:                 scf.yield %49, %44 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:               }
// CHECK-NEXT:               scf.yield %sub_alloc_2, %sub_alloc_3 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:             } else {
// CHECK-NEXT:               scf.yield %elim_a_1, %elim_a1_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:             scf.yield %res_a_1, %res_a1_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %elim_alloc_2, %elim_alloc_3 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %final_alloc_3 : tensor<4x4xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       %50 = func.call @matrix_inverse_4x4(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:       %51 = equivalence.class %52, %50, %final_alloc, %25, %53 : tensor<4x4xf32>
// CHECK-NEXT:       %53 = scf.execute_region -> (tensor<4x4xf32>) {
// CHECK-NEXT:         %c0_3 = arith.constant 0 : index
// CHECK-NEXT:         %c1_3 = arith.constant 1 : index
// CHECK-NEXT:         %c4_3 = arith.constant 4 : index
// CHECK-NEXT:         %cst_6 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:         %cst_7 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:         %alloc_4 = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:         %alloc_5 = equivalence.class %alloc_4 : tensor<4x4xf32>
// CHECK-NEXT:         %init_alloc_6, %init_alloc_7 = scf.for %arg1_5 = %c0_4 to %c4_4 step %c1_4 iter_args(%t_alloc_4 = %alloc_5, %t_alloc_5 = %alloc_5) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:           %new_alloc_4, %new_alloc_5 = scf.for %arg2_6 = %c0_4 to %c4_4 step %c1_4 iter_args(%t_alloc_inner_2 = %t_alloc_4, %t_alloc_1_inner_2 = %t_alloc_5) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:             %54 = arith.cmpi eq, %arg1_5, %arg2_6 : index
// CHECK-NEXT:             %55 = arith.select %54, %cst_8, %cst_9 : f32
// CHECK-NEXT:             %56 = tensor.insert %55 into %t_alloc_inner_2[%arg1_5, %arg2_6] : tensor<4x4xf32>
// CHECK-NEXT:             %57 = tensor.extract %arg0[%arg1_5, %arg2_6] : tensor<4x4xf32>
// CHECK-NEXT:             %58 = tensor.insert %57 into %t_alloc_1_inner_2[%arg1_5, %arg2_6] : tensor<4x4xf32>
// CHECK-NEXT:             scf.yield %56, %58 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %new_alloc_4, %new_alloc_5 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         %final_alloc_5, %final_alloc_6 = scf.for %arg1_6 = %c0_4 to %c4_4 step %c1_4 iter_args(%outer_alloc_4 = %init_alloc_8, %outer_alloc_5 = %init_alloc_9) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:           %59 = tensor.extract %outer_alloc_5[%arg1_6, %arg1_6] : tensor<4x4xf32>
// CHECK-NEXT:           %norm_alloc_4, %norm_alloc_5 = scf.for %arg2_7 = %c0_4 to %c4_4 step %c1_4 iter_args(%norm_a_2 = %outer_alloc_4, %norm_a1_2 = %outer_alloc_5) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:             %60 = tensor.extract %norm_a1_2[%arg1_6, %arg2_7] : tensor<4x4xf32>
// CHECK-NEXT:             %61 = arith.divf %60, %59 : f32
// CHECK-NEXT:             %62 = tensor.insert %61 into %norm_a1_2[%arg1_6, %arg2_7] : tensor<4x4xf32>
// CHECK-NEXT:             %63 = tensor.extract %norm_a_2[%arg1_6, %arg2_7] : tensor<4x4xf32>
// CHECK-NEXT:             %64 = arith.divf %63, %59 : f32
// CHECK-NEXT:             %65 = tensor.insert %64 into %norm_a_2[%arg1_6, %arg2_7] : tensor<4x4xf32>
// CHECK-NEXT:             scf.yield %65, %62 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           %elim_alloc_4, %elim_alloc_5 = scf.for %arg2_8 = %c0_4 to %c4_4 step %c1_4 iter_args(%elim_a_2 = %norm_alloc_4, %elim_a1_2 = %norm_alloc_5) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:             %66 = arith.cmpi ne, %arg2_8, %arg1_6 : index
// CHECK-NEXT:             %res_a_2, %res_a1_2 = scf.if %66 -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:               %67 = tensor.extract %elim_a1_2[%arg2_8, %arg1_6] : tensor<4x4xf32>
// CHECK-NEXT:               %sub_alloc_4, %sub_alloc_5 = scf.for %arg3_2 = %c0_4 to %c4_4 step %c1_4 iter_args(%sub_a_2 = %elim_a_2, %sub_a1_2 = %elim_a1_2) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:                 %68 = tensor.extract %sub_a1_2[%arg1_6, %arg3_2] : tensor<4x4xf32>
// CHECK-NEXT:                 %69 = tensor.extract %sub_a1_2[%arg2_8, %arg3_2] : tensor<4x4xf32>
// CHECK-NEXT:                 %70 = arith.mulf %67, %68 : f32
// CHECK-NEXT:                 %71 = arith.subf %69, %70 : f32
// CHECK-NEXT:                 %72 = tensor.insert %71 into %sub_a1_2[%arg2_8, %arg3_2] : tensor<4x4xf32>
// CHECK-NEXT:                 %73 = tensor.extract %sub_a_2[%arg1_6, %arg3_2] : tensor<4x4xf32>
// CHECK-NEXT:                 %74 = tensor.extract %sub_a_2[%arg2_8, %arg3_2] : tensor<4x4xf32>
// CHECK-NEXT:                 %75 = arith.mulf %67, %73 : f32
// CHECK-NEXT:                 %76 = arith.subf %74, %75 : f32
// CHECK-NEXT:                 %77 = tensor.insert %76 into %sub_a_2[%arg2_8, %arg3_2] : tensor<4x4xf32>
// CHECK-NEXT:                 scf.yield %77, %72 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:               }
// CHECK-NEXT:               scf.yield %sub_alloc_4, %sub_alloc_5 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:             } else {
// CHECK-NEXT:               scf.yield %elim_a_2, %elim_a1_2 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:             scf.yield %res_a_2, %res_a1_2 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %elim_alloc_4, %elim_alloc_5 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         %78 = equivalence.class %79, %final_alloc_5, %80 : tensor<4x4xf32>
// CHECK-NEXT:         %c0_4 = equivalence.class %c0_3 : index
// CHECK-NEXT:         %c1_4 = equivalence.class %c1_3 : index
// CHECK-NEXT:         %c4_4 = equivalence.class %c4_3 : index
// CHECK-NEXT:         %cst_8 = equivalence.class %cst_6 : f32
// CHECK-NEXT:         %cst_9 = equivalence.class %cst_7 : f32
// CHECK-NEXT:         %init_alloc_8 = equivalence.class %init_alloc_6 : tensor<4x4xf32>
// CHECK-NEXT:         %init_alloc_9 = equivalence.class %init_alloc_7 : tensor<4x4xf32>
// CHECK-NEXT:         %final_alloc_7 = equivalence.class %final_alloc_6 : tensor<4x4xf32>
// CHECK-NEXT:         %80 = scf.execute_region -> (tensor<4x4xf32>) {
// CHECK-NEXT:           %c0_5 = arith.constant 0 : index
// CHECK-NEXT:           %c1_5 = arith.constant 1 : index
// CHECK-NEXT:           %c4_5 = arith.constant 4 : index
// CHECK-NEXT:           %cst_10 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:           %cst_11 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:           %alloc_6 = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:           %alloc_7 = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:           %init_alloc_10, %init_alloc_11 = scf.for %arg1_7 = %c0_5 to %c4_5 step %c1_5 iter_args(%t_alloc_6 = %alloc_6, %t_alloc_7 = %alloc_7) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:             %new_alloc_6, %new_alloc_7 = scf.for %arg2_9 = %c0_5 to %c4_5 step %c1_5 iter_args(%t_alloc_inner_3 = %t_alloc_6, %t_alloc_1_inner_3 = %t_alloc_7) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:               %81 = arith.cmpi eq, %arg1_7, %arg2_9 : index
// CHECK-NEXT:               %82 = arith.select %81, %cst_10, %cst_11 : f32
// CHECK-NEXT:               %83 = tensor.insert %82 into %t_alloc_inner_3[%arg1_7, %arg2_9] : tensor<4x4xf32>
// CHECK-NEXT:               %84 = tensor.extract %arg0[%arg1_7, %arg2_9] : tensor<4x4xf32>
// CHECK-NEXT:               %85 = tensor.insert %84 into %t_alloc_1_inner_3[%arg1_7, %arg2_9] : tensor<4x4xf32>
// CHECK-NEXT:               scf.yield %83, %85 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:             scf.yield %new_alloc_6, %new_alloc_7 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           %final_alloc_8, %final_alloc_9 = scf.for %arg1_8 = %c0_5 to %c4_5 step %c1_5 iter_args(%outer_alloc_6 = %init_alloc_10, %outer_alloc_7 = %init_alloc_11) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:             %86 = tensor.extract %outer_alloc_7[%arg1_8, %arg1_8] : tensor<4x4xf32>
// CHECK-NEXT:             %norm_alloc_6, %norm_alloc_7 = scf.for %arg2_10 = %c0_5 to %c4_5 step %c1_5 iter_args(%norm_a_3 = %outer_alloc_6, %norm_a1_3 = %outer_alloc_7) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:               %87 = tensor.extract %norm_a1_3[%arg1_8, %arg2_10] : tensor<4x4xf32>
// CHECK-NEXT:               %88 = arith.divf %87, %86 : f32
// CHECK-NEXT:               %89 = tensor.insert %88 into %norm_a1_3[%arg1_8, %arg2_10] : tensor<4x4xf32>
// CHECK-NEXT:               %90 = tensor.extract %norm_a_3[%arg1_8, %arg2_10] : tensor<4x4xf32>
// CHECK-NEXT:               %91 = arith.divf %90, %86 : f32
// CHECK-NEXT:               %92 = tensor.insert %91 into %norm_a_3[%arg1_8, %arg2_10] : tensor<4x4xf32>
// CHECK-NEXT:               scf.yield %92, %89 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:             %elim_alloc_6, %elim_alloc_7 = scf.for %arg2_11 = %c0_5 to %c4_5 step %c1_5 iter_args(%elim_a_3 = %norm_alloc_6, %elim_a1_3 = %norm_alloc_7) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:               %93 = arith.cmpi ne, %arg2_11, %arg1_8 : index
// CHECK-NEXT:               %res_a_3, %res_a1_3 = scf.if %93 -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:                 %94 = tensor.extract %elim_a1_3[%arg2_11, %arg1_8] : tensor<4x4xf32>
// CHECK-NEXT:                 %sub_alloc_6, %sub_alloc_7 = scf.for %arg3_3 = %c0_5 to %c4_5 step %c1_5 iter_args(%sub_a_3 = %elim_a_3, %sub_a1_3 = %elim_a1_3) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:                   %95 = tensor.extract %sub_a1_3[%arg1_8, %arg3_3] : tensor<4x4xf32>
// CHECK-NEXT:                   %96 = tensor.extract %sub_a1_3[%arg2_11, %arg3_3] : tensor<4x4xf32>
// CHECK-NEXT:                   %97 = arith.mulf %94, %95 : f32
// CHECK-NEXT:                   %98 = arith.subf %96, %97 : f32
// CHECK-NEXT:                   %99 = tensor.insert %98 into %sub_a1_3[%arg2_11, %arg3_3] : tensor<4x4xf32>
// CHECK-NEXT:                   %100 = tensor.extract %sub_a_3[%arg1_8, %arg3_3] : tensor<4x4xf32>
// CHECK-NEXT:                   %101 = tensor.extract %sub_a_3[%arg2_11, %arg3_3] : tensor<4x4xf32>
// CHECK-NEXT:                   %102 = arith.mulf %94, %100 : f32
// CHECK-NEXT:                   %103 = arith.subf %101, %102 : f32
// CHECK-NEXT:                   %104 = tensor.insert %103 into %sub_a_3[%arg2_11, %arg3_3] : tensor<4x4xf32>
// CHECK-NEXT:                   scf.yield %104, %99 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:                 }
// CHECK-NEXT:                 scf.yield %sub_alloc_6, %sub_alloc_7 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:               } else {
// CHECK-NEXT:                 scf.yield %elim_a_3, %elim_a1_3 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:               }
// CHECK-NEXT:               scf.yield %res_a_3, %res_a1_3 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:             scf.yield %elim_alloc_6, %elim_alloc_7 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %final_alloc_8 : tensor<4x4xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         %79 = func.call @matrix_inverse_4x4(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:         scf.yield %78 : tensor<4x4xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       %52 = func.call @get_inv(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:       %105 = func.call @solve(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:       %106 = equivalence.class %107, %105 : tensor<4x4xf32>
// CHECK-NEXT:       %107 = func.call @dot(%51, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:       equivalence.yield %106 : tensor<4x4xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return %0 : tensor<4x4xf32>
// CHECK-NEXT:   }