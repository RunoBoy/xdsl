// RUN: xdsl-opt %s -p ematch-saturate | filecheck ematch_saturate_inv_filecheck.mlir

// This file aims to show that a function can return an E-class, so instead of completely inlining, we need to
// union these E-classes.

// CHECK:func.func private @g() -> i32
// CHECK-NEXT:  func.func @f() -> i32 {
// CHECK-NEXT:    %res = equivalence.graph : () -> i32 {
// CHECK-NEXT:      %x = func.call @g() : () -> i32
// CHECK-NEXT:      %a = equivalence.class %x : i32
// CHECK-NEXT:      equivalence.yield %a : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %res : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @main() -> i32 {
// CHECK-NEXT:    %res = equivalence.graph : () -> i32 {
// CHECK-NEXT:      %x = func.call @g() : () -> i32
// CHECK-NEXT:      %a = equivalence.class %x, %r : i32
// CHECK-NEXT:      %r = func.call @f() : () -> i32
// CHECK-NEXT:      equivalence.yield %a : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %res : i32
// CHECK-NEXT:  }

func.func private @g() -> i32

func.func @f() -> i32 {
    %res = equivalence.graph : () -> i32 {
        %x_2 = func.call @g() : () -> i32
        %a = equivalence.class %x_2 : i32
        equivalence.yield %a : i32
    }

    func.return %res : i32
}

func.func @main() -> i32 {
    %res_1 = equivalence.graph : () -> i32 {
        %r = func.call @f() : () -> i32
        %y = equivalence.class %r : i32
        equivalence.yield %y : i32
    }
    return %res_1 : i32
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
      %function_call = pdl_interp.apply_constraint "get_function_call"(%arg0 : !pdl.operation) : !pdl.operation -> ^bb411, ^bb1
    ^bb411:
      %visibility = pdl_interp.get_attribute "sym_visibility" of %function_call
      %private = pdl_interp.create_attribute "private"
      pdl_interp.are_equal %private, %visibility : !pdl.attribute -> ^bb1, ^bb42
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
      ematch.add_cloned_eclasses of %region
      %region_iterator = pdl_interp_region.region_iterator(%region : !pdl_region.region)
      %inlined_ops = ematch.dedup_region of %region_iterator
      pdl_interp.is_not_null %inlined_ops : !pdl.range<operation> -> ^bb0, ^bb1
    ^bb0:
      %execute_region = pdl_interp_region.create_operation_with_region "scf.execute_region"(%region : !pdl_region.region) -> (%type : !pdl.type)
      %0 = pdl_interp.get_result 0 of %execute_region
      %1 = ematch.get_class_result %0
      %2 = pdl_interp.create_range %1 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %2 : !pdl.range<value>
      pdl_interp.finalize
    ^bb1:
      // Delete the execute_region
      pdl_interp.finalize
    }
  }
