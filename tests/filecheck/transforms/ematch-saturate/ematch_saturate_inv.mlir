 func.func @get_inv(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = equivalence.graph : () -> (tensor<4x4xf32>) {
      %1 = func.call @matrix_inverse_4x4(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
      equivalence.yield %1 : tensor<4x4xf32>
    }
    return %0 : tensor<4x4xf32>
  }

  func.func @matrix_inverse_4x4(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = equivalence.graph : () -> (tensor<4x4xf32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 1.000000e+00 : f32
      %cst_0 = arith.constant 0.000000e+00 : f32

      // 1. Uninitialized pure tensors (no memory side-effects)
      %alloc = tensor.empty() : tensor<4x4xf32>
      %alloc_1 = tensor.empty() : tensor<4x4xf32>

      // 2. Loop 1: Initialize the matrices using iter_args to pass the updated tensors
      %init_alloc, %init_alloc_1 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%t_alloc = %alloc, %t_alloc_1 = %alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
        %new_alloc, %new_alloc_1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%t_alloc_inner = %t_alloc, %t_alloc_1_inner = %t_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {

          %1 = arith.cmpi eq, %arg1, %arg2 : index
          %2 = arith.select %1, %cst, %cst_0 : f32
          %3 = tensor.insert %2 into %t_alloc_inner[%arg1, %arg2] : tensor<4x4xf32>

          %4 = tensor.extract %arg0[%arg1, %arg2] : tensor<4x4xf32>
          %5 = tensor.insert %4 into %t_alloc_1_inner[%arg1, %arg2] : tensor<4x4xf32>

          scf.yield %3, %5 : tensor<4x4xf32>, tensor<4x4xf32>
        }
        scf.yield %new_alloc, %new_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
      }

      // 3. Loop 2: Gauss-Jordan Elimination
      %final_alloc, %final_alloc_1 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%outer_alloc = %init_alloc, %outer_alloc_1 = %init_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
        %1 = tensor.extract %outer_alloc_1[%arg1, %arg1] : tensor<4x4xf32>

        // Normalize the pivot row
        %norm_alloc, %norm_alloc_1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%norm_a = %outer_alloc, %norm_a1 = %outer_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
          %2 = tensor.extract %norm_a1[%arg1, %arg2] : tensor<4x4xf32>
          %3 = arith.divf %2, %1 : f32
          %4 = tensor.insert %3 into %norm_a1[%arg1, %arg2] : tensor<4x4xf32>

          %5 = tensor.extract %norm_a[%arg1, %arg2] : tensor<4x4xf32>
          %6 = arith.divf %5, %1 : f32
          %7 = tensor.insert %6 into %norm_a[%arg1, %arg2] : tensor<4x4xf32>

          scf.yield %7, %4 : tensor<4x4xf32>, tensor<4x4xf32>
        }

        // Eliminate other rows
        %elim_alloc, %elim_alloc_1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%elim_a = %norm_alloc, %elim_a1 = %norm_alloc_1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
          %2 = arith.cmpi ne, %arg2, %arg1 : index

          // scf.if must now return the tensors so they aren't lost if the branch is taken
          %res_a, %res_a1 = scf.if %2 -> (tensor<4x4xf32>, tensor<4x4xf32>) {
            %3 = tensor.extract %elim_a1[%arg2, %arg1] : tensor<4x4xf32>

            %sub_alloc, %sub_alloc_1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%sub_a = %elim_a, %sub_a1 = %elim_a1) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
              %4 = tensor.extract %sub_a1[%arg1, %arg3] : tensor<4x4xf32>
              %5 = tensor.extract %sub_a1[%arg2, %arg3] : tensor<4x4xf32>
              %6 = arith.mulf %3, %4 : f32
              %7 = arith.subf %5, %6 : f32
              %8 = tensor.insert %7 into %sub_a1[%arg2, %arg3] : tensor<4x4xf32>

              %9 = tensor.extract %sub_a[%arg1, %arg3] : tensor<4x4xf32>
              %10 = tensor.extract %sub_a[%arg2, %arg3] : tensor<4x4xf32>
              %11 = arith.mulf %3, %9 : f32
              %12 = arith.subf %10, %11 : f32
              %13 = tensor.insert %12 into %sub_a[%arg2, %arg3] : tensor<4x4xf32>

              scf.yield %13, %8 : tensor<4x4xf32>, tensor<4x4xf32>
            }
            scf.yield %sub_alloc, %sub_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
          } else {
            scf.yield %elim_a, %elim_a1 : tensor<4x4xf32>, tensor<4x4xf32>
          }

          scf.yield %res_a, %res_a1 : tensor<4x4xf32>, tensor<4x4xf32>
        }
        scf.yield %elim_alloc, %elim_alloc_1 : tensor<4x4xf32>, tensor<4x4xf32>
      }

      // No memref.dealloc required.
      equivalence.yield %final_alloc : tensor<4x4xf32>
    }
    return %0 : tensor<4x4xf32>
  }

  func.func private @dot(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32>
  func.func private @solve(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32>

  func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = equivalence.graph : () -> (tensor<4x4xf32>) {
      %1 = func.call @get_inv(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
      %2 = func.call @dot(%1, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>

      // Note: Preserving your original logic here which yielded %1 (the inverse).
      // If you meant to yield the result of the dot product, change to `%2`.
      equivalence.yield %2 : tensor<4x4xf32>
    }
    return %0 : tensor<4x4xf32>
  }

pdl_interp.func @matcher(%arg0 : !pdl.operation) {
      pdl_interp.check_operation_name of %arg0 is "scf.execute_region" -> ^bb31, ^bb40
    ^bb1:
      pdl_interp.finalize
    ^bb31:
      pdl_interp.record_match @rewriters::@execute_region_rewriter(%arg0 : !pdl.operation) : benefit(1) -> ^bb1
    ^bb40:
      pdl_interp.check_operation_name of %arg0 is "func.call" -> ^bb41, ^bb1
    ^bb41:
      %function_call = pdl_interp.apply_constraint "get_function_call"(%arg0 : !pdl.operation) : !pdl.operation -> ^bb39, ^bb1
    ^bb39:
      %visibility = pdl_interp.get_attribute "sym_visibility" of %function_call
      %private = pdl_interp.create_attribute "private"
      pdl_interp.are_equal %private, %visibility : !pdl.attribute -> ^bb401, ^bb42
    ^bb401:
      %name = pdl_interp.get_attribute "sym_name" of %function_call
      %dot = pdl_interp.create_attribute "dot"
      pdl_interp.are_equal %name, %dot : !pdl.attribute -> ^bb50, ^bb1
    ^bb42:
      // get region of function (function body)
      %function_call_original_region = pdl_interp_region.get_region 0 of %function_call : !pdl_region.region
      %function_call_new_region = pdl_interp_region.clone_region(%function_call_original_region : !pdl_region.region)
      // Replace the function arguments with the SSA values from the caller body
      %caller_args = pdl_interp.get_operands of %arg0 : !pdl.range<value>
      pdl_interp.apply_constraint "replace_func_args_with_correct_definitions"(%caller_args, %function_call_new_region : !pdl.range<value>, !pdl_region.region) -> ^bb424, ^bb1
    ^bb424:
      // check if an equivalence graph is present
      %equivalence_graph = pdl_interp_region.get_operation() called "equivalence.graph" 0 of %function_call_new_region
      pdl_interp.is_not_null %equivalence_graph : !pdl.operation -> ^bb421, ^bb1

    ^bb421: // equivalence graph found
      // extract the equivalence graph body
      %equivalence_region = pdl_interp_region.get_region 0 of %equivalence_graph : !pdl_region.region
      %equivalence_region_v2 = pdl_interp_region.clone_region(%equivalence_region : !pdl_region.region)
      // extract the equivalence yield
      %equivalence_yield = pdl_interp_region.get_operation() called "equivalence.yield" 0 of %equivalence_region_v2
      pdl_interp.is_not_null %equivalence_yield : !pdl.operation -> ^bb422, ^bb1
    ^bb422:
      // create a new operation scf.yield to replace the equivalence.yield
      %yield_operand = pdl_interp.get_operand 0 of %equivalence_yield
      %scf_yield = pdl_interp.create_operation "scf.yield"(%yield_operand : !pdl.value)
      %equivalence_region_v3 = pdl_interp_region.insert_op_into_region(%scf_yield : !pdl.operation) of %equivalence_region_v2
      %equivalence_region_v4 = pdl_interp_region.delete_op_from_region(%equivalence_yield : !pdl.operation) of %equivalence_region_v3

      // Create the execute_region where the function body will be executed
      %result = pdl_interp.get_result 0 of %arg0
      %type = pdl_interp.get_value_type of %result : !pdl.type
      pdl_interp.record_match @rewriters::@func_call_rewriter(%arg0, %equivalence_region_v4, %type : !pdl.operation, !pdl_region.region, !pdl.type) : benefit(2) -> ^bb1
    ^bb50:
      %op1 = pdl_interp.get_operand 0 of %arg0
      %op1_eclass = ematch.get_class_vals %op1
      pdl_interp.foreach %op1_node : !pdl.value in %op1_eclass {
        %op1_op = pdl_interp.get_defining_op of %op1_node : !pdl.value
        pdl_interp.is_not_null %op1_op : !pdl.operation -> ^bb51, ^bb52
        ^bb52:
          pdl_interp.continue
        ^bb51:

          pdl_interp.check_operation_name of %op1_op is "func.call" -> ^bb53, ^bb52
        ^bb53:
          %function_call_2 = pdl_interp.apply_constraint "get_function_call"(%op1_op : !pdl.operation) : !pdl.operation -> ^bb54, ^bb52
        ^bb54:
          %potential_name = pdl_interp.get_attribute "sym_name" of %function_call_2
          %matrix_inverse = pdl_interp.create_attribute "matrix_inverse_4x4"
          pdl_interp.are_equal %potential_name, %matrix_inverse : !pdl.attribute -> ^bb55, ^bb52
        ^bb55:
          %op2 = pdl_interp.get_operand 1 of %arg0
          %op2_type = pdl_interp.get_value_type of %op2 : !pdl.type
          pdl_interp.check_type %op2_type is tensor<4x4xf32> -> ^bb56, ^bb52
        ^bb56:
        // Extract the original matrix A (operand 0 of matrix_inverse_4x4)
          %matrix_a = pdl_interp.get_operand 0 of %op1_op

          // Get the result type of the original dot(inv(A), B) call
          %dot_res = pdl_interp.get_result 0 of %arg0
          %dot_res_type = pdl_interp.get_value_type of %dot_res : !pdl.type

          // Record the match and pass A, B, and the return type to the rewriter
          // NOTE: We branch back to ^bb52 to continue checking other e-nodes in the loop!
          pdl_interp.record_match @rewriters::@solve(%arg0, %matrix_a, %op2, %dot_res_type : !pdl.operation, !pdl.value, !pdl.value, !pdl.type) : benefit(2) -> ^bb52
      } -> ^bb1
   }

  builtin.module @rewriters {
    pdl_interp.func @if_true_rewriter(%arg0 : !pdl.operation, %arg1 : !pdl.type) {
      %0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%0 : !pdl_region.region) -> (%arg1 : !pdl.type)
      %11 = ematch.dedup %1
      %2 = pdl_interp.get_result 0 of %11
      %3 = ematch.get_class_result %2
      %4 = pdl_interp.create_range %3 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %4 : !pdl.range<value>
      pdl_interp.finalize
    }

     pdl_interp.func @if_false_rewriter(%arg0 : !pdl.operation, %arg1 : !pdl.type) {
      %0 = pdl_interp_region.get_region 1 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%0 : !pdl_region.region) -> (%arg1 : !pdl.type)
      %11 = ematch.dedup %1
      %2 = pdl_interp.get_result 0 of %11
      %3 = ematch.get_class_result %2
      %4 = pdl_interp.create_range %3 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %4 : !pdl.range<value>
      pdl_interp.finalize
    }

     pdl_interp.func @execute_region_rewriter(%arg0: !pdl.operation) {
      %0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %1, %inlined_ops = pdl_interp_region.inline_region %arg0 with (%0 : !pdl_region.region)
      %2 = pdl_interp.get_defining_op of %1 : !pdl.value
      pdl_interp.check_operation_name of %2 is "equivalence.class" -> ^bb1, ^bb2
      ^bb1:
          // 1. Capture the yielded class BEFORE anything is erased
          %yielded_class_val = ematch.get_class_result %1
          %yielded_class_range = pdl_interp.create_range %yielded_class_val : !pdl.value

          // 2. Capture the target class (%y) using get_defining_op
          %arg0_res = pdl_interp.get_result 0 of %arg0
          %y_val = ematch.get_class_result %arg0_res
          %y_op = pdl_interp.get_defining_op of %y_val : !pdl.value

          // 3. Forward uses safely so we don't create zombie pointers
          pdl_interp.replace %arg0 with (%1 : !pdl.value)

          // 4. Hashcons the region (Phase 1 of your FSM)
          ematch.dedup_region %inlined_ops

          // 5. Link the E-classes (Phase 2 of your FSM)
          ematch.union %y_op : !pdl.operation, %yielded_class_range : !pdl.range<value>

          pdl_interp.finalize
       ^bb2:
          pdl_interp.replace %arg0 with (%1 : !pdl.value)
          ematch.dedup_region %inlined_ops
          pdl_interp.finalize
    }

    pdl_interp.func @func_call_rewriter(%arg0 : !pdl.operation, %region : !pdl_region.region, %type : !pdl.type) {
      %execute_region = pdl_interp_region.create_operation_with_region "scf.execute_region"(%region : !pdl_region.region) -> (%type : !pdl.type)
      ematch.add_cloned_eclasses of %region
      %0 = pdl_interp.get_result 0 of %execute_region
      %1 = ematch.get_class_result %0
      %2 = pdl_interp.create_range %1 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %2 : !pdl.range<value>
      pdl_interp.finalize
    }

    pdl_interp.func @solve(%arg0 : !pdl.operation, %matrix_a : !pdl.value, %matrix_b : !pdl.value, %res_type : !pdl.type) {
      // 1. Create the @solve symbol attribute
      %callee = "pdl_interp.create_attribute"() <{value = @solve}> : () -> !pdl.attribute

      // 2. Create the `func.call @solve(A, B)` operation
      // We pass 2 operands, 1 attribute, and 1 result type -> operandSegmentSizes: 2, 1, 1
      %new_call = "pdl_interp.create_operation"(%matrix_a, %matrix_b, %callee, %res_type) <{
          name = "func.call",
          inputAttributeNames = ["callee"],
          operandSegmentSizes = array<i32: 2, 1, 1>
      }> : (!pdl.value, !pdl.value, !pdl.attribute, !pdl.type) -> !pdl.operation

      // 3. Deduplicate the new operation in the Hashcons / E-Graph
      %dedup_call = ematch.dedup %new_call

      // 4. Extract the result and fetch its corresponding E-class
      %new_res = pdl_interp.get_result 0 of %dedup_call
      %eclass_res = ematch.get_class_result %new_res
      %res_range = pdl_interp.create_range %eclass_res : !pdl.value

      // 5. Union the original `dot` call with our new `solve` call
      ematch.union %arg0 : !pdl.operation, %res_range : !pdl.range<value>

      pdl_interp.finalize
    }
  }