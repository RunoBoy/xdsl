// RUN: xdsl-opt %s -p ematch-saturate | filecheck ematch_saturate_inv_filecheck.mlir

// This file aims to show that a function can contain an E-class, so when inlining, the rebuild step needs to
// union these E-classes.

// CHECK: func.func private @g() -> i32
// CHECK-NEXT:   func.func private @h(i32) -> i32
// CHECK-NEXT:   func.func @f() -> i32 {
// CHECK-NEXT:     %res = equivalence.graph : () -> i32 {
// CHECK-NEXT:       %x = func.call @g() : () -> i32
// CHECK-NEXT:       %a = equivalence.class %x : i32
// CHECK-NEXT:       %b = func.call @h(%a) : (i32) -> i32
// CHECK-NEXT:       equivalence.yield %b : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return %res : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @main() -> i32 {
// CHECK-NEXT:     %res = equivalence.graph : () -> i32 {
// CHECK-NEXT:       %x = func.call @g() : () -> i32
// CHECK-NEXT:       %a = equivalence.class %x : i32
// CHECK-NEXT:       %b = func.call @h(%a) : (i32) -> i32
// CHECK-NEXT:       %r = equivalence.class %r_1, %b : i32
// CHECK-NEXT:       %r_1 = func.call @f() : () -> i32
// CHECK-NEXT:       equivalence.yield %a : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return %res : i32
// CHECK-NEXT:   }


func.func private @g() -> i32
func.func private @h(%arg0: i32) -> i32

func.func @f() -> i32 {
    %res = equivalence.graph : () -> i32 {
        // x' = g()
        %x_2 = func.call @g() : () -> i32

        // a = E-class(x')
        %a = equivalence.class %x_2 : i32

        // b = h(a)
        %b = func.call @h(%a) : (i32) -> i32

        equivalence.yield %b : i32
    }

    func.return %res : i32
}

func.func @main() -> i32 {
    %res_1 = equivalence.graph : () -> i32 {
        // x = g()
        %x = func.call @g() : () -> i32

        // r = call f(...)
        %r = func.call @f() : () -> i32

        // y = E-class(x)
        %y = equivalence.class %x : i32

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
      pdl_interp.check_operation_name of %2 is "equivalence.class" -> ^bb0, ^bb1
    // If an E-class is returned, we union the classes with the original execute_region E-class
    ^bb0:
      // Extract the E-class of the value that was just yielded by the inlined region
      %new_eclass_val = ematch.get_class_result %1
      %new_eclass_range = pdl_interp.create_range %new_eclass_val : !pdl.value

      // Grab the result of the original execute_region
      %original_res = pdl_interp.get_result 0 of %arg0
      %original_eclass_val = ematch.get_class_result %original_res
      // Now grab the E-class belonging to the original execute_region
      %original_eclass_op = pdl_interp.get_defining_op of %original_eclass_val : !pdl.value

      // Replace uses of the old operation with the newly inlined value
      pdl_interp.replace %arg0 with (%1 : !pdl.value)

      // Deduplicate the region and union the E-classes
      ematch.dedup_region %inlined_ops
      ematch.union %original_eclass_op : !pdl.operation, %new_eclass_range : !pdl.range<value>

      pdl_interp.finalize
    // If a regular operation is returned, we only deduplicate the region
    ^bb1:
      pdl_interp.replace %arg0 with (%1 : !pdl.value)
      ematch.dedup_region of %inlined_ops
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
