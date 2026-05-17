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
// CHECK-NEXT:      %alloc = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:      %alloc_1 = equivalence.class %alloc_2 : tensor<4x4xf32>
// CHECK-NEXT:      %alloc_2 = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:      %init_alloc, %init_alloc_1 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%t_alloc = %alloc, %t_alloc_1 = %alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:        %new_alloc, %new_alloc_1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%t_alloc_inner = %t_alloc, %t_alloc_1_inner = %t_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %1 = arith.cmpi eq, %arg1, %arg2 : index
// CHECK-NEXT:          %2 = arith.select %1, %cst, %cst_2 : f32
// CHECK-NEXT:          %3 = tensor.insert %2 into %t_alloc_inner[%arg1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:          %4 = tensor.extract %arg0[%arg1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:          %5 = tensor.insert %4 into %t_alloc_1_inner[%arg1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:          scf.yield %3, %5 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %new_alloc, %new_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %final_alloc, %final_alloc_1 = scf.for %arg1_1 = %c0 to %c4 step %c1 iter_args(%outer_alloc = %init_alloc, %outer_alloc_1 = %init_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:        %6 = tensor.extract %outer_alloc_1[%arg1_1, %arg1_1] : tensor<4x4xf32>
// CHECK-NEXT:        %norm_alloc, %norm_alloc_1 = scf.for %arg2_1 = %c0 to %c4 step %c1 iter_args(%norm_a = %outer_alloc, %norm_a1 = %outer_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %7 = tensor.extract %norm_a1[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          %8 = arith.divf %7, %6 : f32
// CHECK-NEXT:          %9 = tensor.insert %8 into %norm_a1[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          %10 = tensor.extract %norm_a[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          %11 = arith.divf %10, %6 : f32
// CHECK-NEXT:          %12 = tensor.insert %11 into %norm_a[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          scf.yield %12, %9 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %elim_alloc, %elim_alloc_1 = scf.for %arg2_2 = %c0 to %c4 step %c1 iter_args(%elim_a = %norm_alloc, %elim_a1 = %norm_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %13 = arith.cmpi ne, %arg2_2, %arg1_1 : index
// CHECK-NEXT:          %res_a, %res_a1 = scf.if %13 -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:            %14 = tensor.extract %elim_a1[%arg2_2, %arg1_1] : tensor<4x4xf32>
// CHECK-NEXT:            %sub_alloc, %sub_alloc_1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%sub_a = %elim_a, %sub_a1 = %elim_a1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:              %15 = tensor.extract %sub_a1[%arg1_1, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %16 = tensor.extract %sub_a1[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %17 = arith.mulf %14, %15 : f32
// CHECK-NEXT:              %18 = arith.subf %16, %17 : f32
// CHECK-NEXT:              %19 = tensor.insert %18 into %sub_a1[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %20 = tensor.extract %sub_a[%arg1_1, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %21 = tensor.extract %sub_a[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %22 = arith.mulf %14, %20 : f32
// CHECK-NEXT:              %23 = arith.subf %21, %22 : f32
// CHECK-NEXT:              %24 = tensor.insert %23 into %sub_a[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              scf.yield %24, %19 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %sub_alloc, %sub_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:          } else {
// CHECK-NEXT:            scf.yield %elim_a, %elim_a1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield %res_a, %res_a1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %elim_alloc, %elim_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %final_alloc_2 = equivalence.class %25, %final_alloc, %final_alloc_3 : tensor<4x4xf32>
// CHECK-NEXT:      %init_alloc_2 = equivalence.class %init_alloc_3 : tensor<4x4xf32>
// CHECK-NEXT:      %init_alloc_4 = equivalence.class %init_alloc_5 : tensor<4x4xf32>
// CHECK-NEXT:      %init_alloc_3, %init_alloc_5 = scf.for %arg1_2 = %c0 to %c4 step %c1 iter_args(%t_alloc_2 = %alloc_1, %t_alloc_3 = %alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:        %new_alloc_2, %new_alloc_3 = scf.for %arg2_3 = %c0 to %c4 step %c1 iter_args(%t_alloc_inner_1 = %t_alloc_2, %t_alloc_1_inner_1 = %t_alloc_3) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %26 = arith.cmpi eq, %arg1_2, %arg2_3 : index
// CHECK-NEXT:          %27 = arith.select %26, %cst, %cst_2 : f32
// CHECK-NEXT:          %28 = tensor.insert %27 into %t_alloc_inner_1[%arg1_2, %arg2_3] : tensor<4x4xf32>
// CHECK-NEXT:          %29 = tensor.extract %arg0[%arg1_2, %arg2_3] : tensor<4x4xf32>
// CHECK-NEXT:          %30 = tensor.insert %29 into %t_alloc_1_inner_1[%arg1_2, %arg2_3] : tensor<4x4xf32>
// CHECK-NEXT:          scf.yield %28, %30 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %new_alloc_2, %new_alloc_3 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %final_alloc_4 = equivalence.class %final_alloc_5 : tensor<4x4xf32>
// CHECK-NEXT:      %final_alloc_3, %final_alloc_5 = scf.for %arg1_3 = %c0 to %c4 step %c1 iter_args(%outer_alloc_2 = %init_alloc_2, %outer_alloc_3 = %init_alloc_4) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:        %31 = tensor.extract %outer_alloc_3[%arg1_3, %arg1_3] : tensor<4x4xf32>
// CHECK-NEXT:        %norm_alloc_2, %norm_alloc_3 = scf.for %arg2_4 = %c0 to %c4 step %c1 iter_args(%norm_a_1 = %outer_alloc_2, %norm_a1_1 = %outer_alloc_3) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %32 = tensor.extract %norm_a1_1[%arg1_3, %arg2_4] : tensor<4x4xf32>
// CHECK-NEXT:          %33 = arith.divf %32, %31 : f32
// CHECK-NEXT:          %34 = tensor.insert %33 into %norm_a1_1[%arg1_3, %arg2_4] : tensor<4x4xf32>
// CHECK-NEXT:          %35 = tensor.extract %norm_a_1[%arg1_3, %arg2_4] : tensor<4x4xf32>
// CHECK-NEXT:          %36 = arith.divf %35, %31 : f32
// CHECK-NEXT:          %37 = tensor.insert %36 into %norm_a_1[%arg1_3, %arg2_4] : tensor<4x4xf32>
// CHECK-NEXT:          scf.yield %37, %34 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %elim_alloc_2, %elim_alloc_3 = scf.for %arg2_5 = %c0 to %c4 step %c1 iter_args(%elim_a_1 = %norm_alloc_2, %elim_a1_1 = %norm_alloc_3) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %38 = arith.cmpi ne, %arg2_5, %arg1_3 : index
// CHECK-NEXT:          %res_a_1, %res_a1_1 = scf.if %38 -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:            %39 = tensor.extract %elim_a1_1[%arg2_5, %arg1_3] : tensor<4x4xf32>
// CHECK-NEXT:            %sub_alloc_2, %sub_alloc_3 = scf.for %arg3_1 = %c0 to %c4 step %c1 iter_args(%sub_a_1 = %elim_a_1, %sub_a1_1 = %elim_a1_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:              %40 = tensor.extract %sub_a1_1[%arg1_3, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:              %41 = tensor.extract %sub_a1_1[%arg2_5, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:              %42 = arith.mulf %39, %40 : f32
// CHECK-NEXT:              %43 = arith.subf %41, %42 : f32
// CHECK-NEXT:              %44 = tensor.insert %43 into %sub_a1_1[%arg2_5, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:              %45 = tensor.extract %sub_a_1[%arg1_3, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:              %46 = tensor.extract %sub_a_1[%arg2_5, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:              %47 = arith.mulf %39, %45 : f32
// CHECK-NEXT:              %48 = arith.subf %46, %47 : f32
// CHECK-NEXT:              %49 = tensor.insert %48 into %sub_a_1[%arg2_5, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:              scf.yield %49, %44 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %sub_alloc_2, %sub_alloc_3 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:          } else {
// CHECK-NEXT:            scf.yield %elim_a_1, %elim_a1_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield %res_a_1, %res_a1_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %elim_alloc_2, %elim_alloc_3 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %25 = func.call @matrix_inverse_4x4(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:      equivalence.yield %final_alloc_2 : tensor<4x4xf32>
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
// CHECK-NEXT:      %alloc_1 = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:      %init_alloc, %init_alloc_1 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%t_alloc = %alloc, %t_alloc_1 = %alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:        %new_alloc, %new_alloc_1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%t_alloc_inner = %t_alloc, %t_alloc_1_inner = %t_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %1 = arith.cmpi eq, %arg1, %arg2 : index
// CHECK-NEXT:          %2 = arith.select %1, %cst, %cst_1 : f32
// CHECK-NEXT:          %3 = tensor.insert %2 into %t_alloc_inner[%arg1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:          %4 = tensor.extract %arg0[%arg1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:          %5 = tensor.insert %4 into %t_alloc_1_inner[%arg1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:          scf.yield %3, %5 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %new_alloc, %new_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %final_alloc, %final_alloc_1 = scf.for %arg1_1 = %c0 to %c4 step %c1 iter_args(%outer_alloc = %init_alloc, %outer_alloc_1 = %init_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:        %6 = tensor.extract %outer_alloc_1[%arg1_1, %arg1_1] : tensor<4x4xf32>
// CHECK-NEXT:        %norm_alloc, %norm_alloc_1 = scf.for %arg2_1 = %c0 to %c4 step %c1 iter_args(%norm_a = %outer_alloc, %norm_a1 = %outer_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %7 = tensor.extract %norm_a1[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          %8 = arith.divf %7, %6 : f32
// CHECK-NEXT:          %9 = tensor.insert %8 into %norm_a1[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          %10 = tensor.extract %norm_a[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          %11 = arith.divf %10, %6 : f32
// CHECK-NEXT:          %12 = tensor.insert %11 into %norm_a[%arg1_1, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          scf.yield %12, %9 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %elim_alloc, %elim_alloc_1 = scf.for %arg2_2 = %c0 to %c4 step %c1 iter_args(%elim_a = %norm_alloc, %elim_a1 = %norm_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %13 = arith.cmpi ne, %arg2_2, %arg1_1 : index
// CHECK-NEXT:          %res_a, %res_a1 = scf.if %13 -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:            %14 = tensor.extract %elim_a1[%arg2_2, %arg1_1] : tensor<4x4xf32>
// CHECK-NEXT:            %sub_alloc, %sub_alloc_1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%sub_a = %elim_a, %sub_a1 = %elim_a1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:              %15 = tensor.extract %sub_a1[%arg1_1, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %16 = tensor.extract %sub_a1[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %17 = arith.mulf %14, %15 : f32
// CHECK-NEXT:              %18 = arith.subf %16, %17 : f32
// CHECK-NEXT:              %19 = tensor.insert %18 into %sub_a1[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %20 = tensor.extract %sub_a[%arg1_1, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %21 = tensor.extract %sub_a[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %22 = arith.mulf %14, %20 : f32
// CHECK-NEXT:              %23 = arith.subf %21, %22 : f32
// CHECK-NEXT:              %24 = tensor.insert %23 into %sub_a[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              scf.yield %24, %19 : tensor<4x4xf32>, tensor<4x4xf32>
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
// CHECK-NEXT:      %alloc = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:      %alloc_1 = equivalence.class %alloc_2 : tensor<4x4xf32>
// CHECK-NEXT:      %alloc_2 = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:      %init_alloc, %init_alloc_1 = scf.for %arg1_1 = %c0 to %c4 step %c1 iter_args(%t_alloc = %alloc, %t_alloc_1 = %alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:        %new_alloc, %new_alloc_1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%t_alloc_inner = %t_alloc, %t_alloc_1_inner = %t_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %1 = arith.cmpi eq, %arg1_1, %arg2 : index
// CHECK-NEXT:          %2 = arith.select %1, %cst, %cst_2 : f32
// CHECK-NEXT:          %3 = tensor.insert %2 into %t_alloc_inner[%arg1_1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:          %4 = tensor.extract %arg0[%arg1_1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:          %5 = tensor.insert %4 into %t_alloc_1_inner[%arg1_1, %arg2] : tensor<4x4xf32>
// CHECK-NEXT:          scf.yield %3, %5 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %new_alloc, %new_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %final_alloc, %final_alloc_1 = scf.for %arg1_2 = %c0 to %c4 step %c1 iter_args(%outer_alloc = %init_alloc, %outer_alloc_1 = %init_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:        %6 = tensor.extract %outer_alloc_1[%arg1_2, %arg1_2] : tensor<4x4xf32>
// CHECK-NEXT:        %norm_alloc, %norm_alloc_1 = scf.for %arg2_1 = %c0 to %c4 step %c1 iter_args(%norm_a = %outer_alloc, %norm_a1 = %outer_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %7 = tensor.extract %norm_a1[%arg1_2, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          %8 = arith.divf %7, %6 : f32
// CHECK-NEXT:          %9 = tensor.insert %8 into %norm_a1[%arg1_2, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          %10 = tensor.extract %norm_a[%arg1_2, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          %11 = arith.divf %10, %6 : f32
// CHECK-NEXT:          %12 = tensor.insert %11 into %norm_a[%arg1_2, %arg2_1] : tensor<4x4xf32>
// CHECK-NEXT:          scf.yield %12, %9 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %elim_alloc, %elim_alloc_1 = scf.for %arg2_2 = %c0 to %c4 step %c1 iter_args(%elim_a = %norm_alloc, %elim_a1 = %norm_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %13 = arith.cmpi ne, %arg2_2, %arg1_2 : index
// CHECK-NEXT:          %res_a, %res_a1 = scf.if %13 -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:            %14 = tensor.extract %elim_a1[%arg2_2, %arg1_2] : tensor<4x4xf32>
// CHECK-NEXT:            %sub_alloc, %sub_alloc_1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%sub_a = %elim_a, %sub_a1 = %elim_a1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:              %15 = tensor.extract %sub_a1[%arg1_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %16 = tensor.extract %sub_a1[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %17 = arith.mulf %14, %15 : f32
// CHECK-NEXT:              %18 = arith.subf %16, %17 : f32
// CHECK-NEXT:              %19 = tensor.insert %18 into %sub_a1[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %20 = tensor.extract %sub_a[%arg1_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %21 = tensor.extract %sub_a[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              %22 = arith.mulf %14, %20 : f32
// CHECK-NEXT:              %23 = arith.subf %21, %22 : f32
// CHECK-NEXT:              %24 = tensor.insert %23 into %sub_a[%arg2_2, %arg3] : tensor<4x4xf32>
// CHECK-NEXT:              scf.yield %24, %19 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %sub_alloc, %sub_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:          } else {
// CHECK-NEXT:            scf.yield %elim_a, %elim_a1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield %res_a, %res_a1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %elim_alloc, %elim_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %init_alloc_2 = equivalence.class %init_alloc_3 : tensor<4x4xf32>
// CHECK-NEXT:      %init_alloc_4 = equivalence.class %init_alloc_5 : tensor<4x4xf32>
// CHECK-NEXT:      %init_alloc_3, %init_alloc_5 = scf.for %arg1_3 = %c0 to %c4 step %c1 iter_args(%t_alloc_2 = %alloc_1, %t_alloc_3 = %alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:        %new_alloc_2, %new_alloc_3 = scf.for %arg2_3 = %c0 to %c4 step %c1 iter_args(%t_alloc_inner_1 = %t_alloc_2, %t_alloc_1_inner_1 = %t_alloc_3) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %25 = arith.cmpi eq, %arg1_3, %arg2_3 : index
// CHECK-NEXT:          %26 = arith.select %25, %cst, %cst_2 : f32
// CHECK-NEXT:          %27 = tensor.insert %26 into %t_alloc_inner_1[%arg1_3, %arg2_3] : tensor<4x4xf32>
// CHECK-NEXT:          %28 = tensor.extract %arg0[%arg1_3, %arg2_3] : tensor<4x4xf32>
// CHECK-NEXT:          %29 = tensor.insert %28 into %t_alloc_1_inner_1[%arg1_3, %arg2_3] : tensor<4x4xf32>
// CHECK-NEXT:          scf.yield %27, %29 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %new_alloc_2, %new_alloc_3 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %final_alloc_2 = equivalence.class %final_alloc_3 : tensor<4x4xf32>
// CHECK-NEXT:      %final_alloc_4, %final_alloc_3 = scf.for %arg1_4 = %c0 to %c4 step %c1 iter_args(%outer_alloc_2 = %init_alloc_2, %outer_alloc_3 = %init_alloc_4) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:        %30 = tensor.extract %outer_alloc_3[%arg1_4, %arg1_4] : tensor<4x4xf32>
// CHECK-NEXT:        %norm_alloc_2, %norm_alloc_3 = scf.for %arg2_4 = %c0 to %c4 step %c1 iter_args(%norm_a_1 = %outer_alloc_2, %norm_a1_1 = %outer_alloc_3) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %31 = tensor.extract %norm_a1_1[%arg1_4, %arg2_4] : tensor<4x4xf32>
// CHECK-NEXT:          %32 = arith.divf %31, %30 : f32
// CHECK-NEXT:          %33 = tensor.insert %32 into %norm_a1_1[%arg1_4, %arg2_4] : tensor<4x4xf32>
// CHECK-NEXT:          %34 = tensor.extract %norm_a_1[%arg1_4, %arg2_4] : tensor<4x4xf32>
// CHECK-NEXT:          %35 = arith.divf %34, %30 : f32
// CHECK-NEXT:          %36 = tensor.insert %35 into %norm_a_1[%arg1_4, %arg2_4] : tensor<4x4xf32>
// CHECK-NEXT:          scf.yield %36, %33 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %elim_alloc_2, %elim_alloc_3 = scf.for %arg2_5 = %c0 to %c4 step %c1 iter_args(%elim_a_1 = %norm_alloc_2, %elim_a1_1 = %norm_alloc_3) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:          %37 = arith.cmpi ne, %arg2_5, %arg1_4 : index
// CHECK-NEXT:          %res_a_1, %res_a1_1 = scf.if %37 -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:            %38 = tensor.extract %elim_a1_1[%arg2_5, %arg1_4] : tensor<4x4xf32>
// CHECK-NEXT:            %sub_alloc_2, %sub_alloc_3 = scf.for %arg3_1 = %c0 to %c4 step %c1 iter_args(%sub_a_1 = %elim_a_1, %sub_a1_1 = %elim_a1_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:              %39 = tensor.extract %sub_a1_1[%arg1_4, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:              %40 = tensor.extract %sub_a1_1[%arg2_5, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:              %41 = arith.mulf %38, %39 : f32
// CHECK-NEXT:              %42 = arith.subf %40, %41 : f32
// CHECK-NEXT:              %43 = tensor.insert %42 into %sub_a1_1[%arg2_5, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:              %44 = tensor.extract %sub_a_1[%arg1_4, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:              %45 = tensor.extract %sub_a_1[%arg2_5, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:              %46 = arith.mulf %38, %44 : f32
// CHECK-NEXT:              %47 = arith.subf %45, %46 : f32
// CHECK-NEXT:              %48 = tensor.insert %47 into %sub_a_1[%arg2_5, %arg3_1] : tensor<4x4xf32>
// CHECK-NEXT:              scf.yield %48, %43 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %sub_alloc_2, %sub_alloc_3 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:          } else {
// CHECK-NEXT:            scf.yield %elim_a_1, %elim_a1_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield %res_a_1, %res_a1_1 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %elim_alloc_2, %elim_alloc_3 : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %49 = func.call @matrix_inverse_4x4(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:      %final_alloc_5 = equivalence.class %50, %49, %final_alloc, %final_alloc_4 : tensor<4x4xf32>
// CHECK-NEXT:      %50 = func.call @get_inv(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:      %51 = func.call @solve(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:      %52 = equivalence.class %53, %51 : tensor<4x4xf32>
// CHECK-NEXT:      %53 = func.call @dot(%final_alloc_5, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:      equivalence.yield %52 : tensor<4x4xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %0 : tensor<4x4xf32>
// CHECK-NEXT:  }