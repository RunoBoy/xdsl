// xdsl-opt %x -p ematch-saturate | filecheck %s

// CHECK: func.func private @h() -> i32
// CHECK-NEXT:   func.func @f() -> i32 {
// CHECK-NEXT:     %res = equivalence.graph : () -> i32 {
// CHECK-NEXT:       %cond = arith.constant false
// CHECK-NEXT:       %x = scf.if %cond -> (i32) {
// CHECK-NEXT:         %a = arith.constant 2 : i32
// CHECK-NEXT:         scf.yield %a : i32
// CHECK-NEXT:       } else {
// CHECK-NEXT:         %b = arith.constant 3 : i32
// CHECK-NEXT:         scf.yield %b : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       %vv = func.call @h() : () -> i32
// CHECK-NEXT:       %c = equivalence.class %vv, %x : i32
// CHECK-NEXT:       equivalence.yield %c : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return %res : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @main() -> i32 {
// CHECK-NEXT:     %res = equivalence.graph : () -> i32 {
// CHECK-NEXT:       %x = func.call @h() : () -> i32
// CHECK-NEXT:       %cond = equivalence.class %cond_1 : i1
// CHECK-NEXT:       %cond_1 = arith.constant false
// CHECK-NEXT:       %x_1 = scf.if %cond -> (i32) {
// CHECK-NEXT:         %a = arith.constant 2 : i32
// CHECK-NEXT:         scf.yield %a : i32
// CHECK-NEXT:       } else {
// CHECK-NEXT:         %b = arith.constant 3 : i32
// CHECK-NEXT:         scf.yield %b : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       %r = func.call @f() : () -> i32
// CHECK-NEXT:       %y = equivalence.class %x, %r, %x_1 : i32
// CHECK-NEXT:       equivalence.yield %y : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return %res : i32
// CHECK-NEXT:   }

func.func private @h() -> i32
func.func @f() -> i32 {
    %res = equivalence.graph : () -> i32 {
        %cond = arith.constant 0 : i1
        %x = scf.if %cond -> (i32) {
            %a = arith.constant 2 : i32
            scf.yield %a : i32
        } else {
            %b = arith.constant 3 : i32
            scf.yield %b : i32
        }

        %vv = func.call @h() : () -> i32

        %c = equivalence.class %vv, %x : i32

        equivalence.yield %c : i32
    }

    func.return %res : i32
}

func.func @main() -> i32 {
    %res_1 = equivalence.graph : () -> i32 {
        %x = func.call @h() : () -> i32
        %r = func.call @f() : () -> i32

        %y = equivalence.class %x, %r : i32

        equivalence.yield %y : i32
    }

    return %res_1 : i32
}

  pdl_interp.func @matcher(%arg0: !pdl.operation) {
    pdl_interp.check_operation_name of %arg0 is "scf.execute_region" -> ^bb0, ^bb1
  ^bb2:
    pdl_interp.finalize
  ^bb0:
    pdl_interp.record_match @rewriters::@execute_region_rewriter(%arg0 : !pdl.operation) : benefit(1), loc([]) -> ^bb2
  ^bb1:
    pdl_interp.check_operation_name of %arg0 is "func.call" -> ^bb3, ^bb2
  ^bb3:
    %function_call = pdl_interp.apply_constraint "get_function_call"(%arg0 : !pdl.operation) : !pdl.operation -> ^bb4, ^bb2
  ^bb4:
    %visibility = pdl_interp.get_attribute "sym_visibility" of %function_call
    %private = pdl_interp.create_attribute "private"
    pdl_interp.are_equal %private, %visibility : !pdl.attribute -> ^bb2, ^bb5
  ^bb5:
    %function_call_original_region = pdl_interp_region.get_region 0 of %function_call : !pdl_region.region
    %function_call_new_region = pdl_interp_region.clone_region(%function_call_original_region : !pdl_region.region)
    %caller_args = pdl_interp.get_operands of %arg0 : !pdl.range<value>
    pdl_interp.apply_constraint "replace_func_args_with_correct_definitions"(%caller_args, %function_call_new_region : !pdl.range<value>, !pdl_region.region) -> ^bb6, ^bb2
  ^bb6:
    %equivalence_graph = pdl_interp_region.get_operation() called "equivalence.graph"  0 of %function_call_new_region
    pdl_interp.is_not_null %equivalence_graph : !pdl.operation -> ^bb7, ^bb2
  ^bb7:
    %equivalence_region = pdl_interp_region.get_region 0 of %equivalence_graph : !pdl_region.region
    %equivalence_region_v2 = pdl_interp_region.clone_region(%equivalence_region : !pdl_region.region)
    %equivalence_yield = pdl_interp_region.get_operation() called "equivalence.yield"  0 of %equivalence_region_v2
    pdl_interp.is_not_null %equivalence_yield : !pdl.operation -> ^bb8, ^bb2
  ^bb8:
    %yield_operand = pdl_interp.get_operand 0 of %equivalence_yield
    %scf_yield = pdl_interp.create_operation "scf.yield"(%yield_operand : !pdl.value)
    %equivalence_region_v3 = pdl_interp_region.insert_op_into_region(%scf_yield : !pdl.operation) of %equivalence_region_v2
    %equivalence_region_v4 = pdl_interp_region.delete_op_from_region(%equivalence_yield : !pdl.operation) of %equivalence_region_v3
    %result = pdl_interp.get_result 0 of %arg0
    %type = pdl_interp.get_value_type of %result : !pdl.type
    pdl_interp.record_match @rewriters::@func_call_rewriter(%arg0, %equivalence_region_v4, %type : !pdl.operation, !pdl_region.region, !pdl.type) : benefit(2), loc([]) -> ^bb2
  }
  builtin.module @rewriters {
    pdl_interp.func @if_true_rewriter(%arg0: !pdl.operation, %arg1: !pdl.type) {
      %0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%0 : !pdl_region.region) -> (%arg1 : !pdl.type)
      %2 = ematch.dedup %1
      %3 = pdl_interp.get_result 0 of %2
      %4 = ematch.get_class_result %3
      %5 = pdl_interp.create_range %4 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %5 : !pdl.range<value>
      pdl_interp.finalize
    }
    pdl_interp.func @if_false_rewriter(%arg0: !pdl.operation, %arg1: !pdl.type) {
      %0 = pdl_interp_region.get_region 1 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%0 : !pdl_region.region) -> (%arg1 : !pdl.type)
      %2 = ematch.dedup %1
      %3 = pdl_interp.get_result 0 of %2
      %4 = ematch.get_class_result %3
      %5 = pdl_interp.create_range %4 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %5 : !pdl.range<value>
      pdl_interp.finalize
    }
    pdl_interp.func @execute_region_rewriter(%arg0: !pdl.operation) {
      %0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %1, %inlined_ops = pdl_interp_region.inline_region %arg0 with (%0 : !pdl_region.region)
      %2 = pdl_interp.get_defining_op of %1 : !pdl.value
      pdl_interp.check_operation_name of %2 is "equivalence.class" -> ^bb0, ^bb1
    ^bb0:
      %new_eclass_val = ematch.get_class_result %1
      %new_eclass_range = pdl_interp.create_range %new_eclass_val : !pdl.value
      %original_res = pdl_interp.get_result 0 of %arg0
      %original_eclass_val = ematch.get_class_result %original_res
      %original_eclass_op = pdl_interp.get_defining_op of %original_eclass_val : !pdl.value
      pdl_interp.replace %arg0 with (%1 : !pdl.value)
      %3 = ematch.dedup_region of %inlined_ops
      ematch.union %original_eclass_op : !pdl.operation, %new_eclass_range : !pdl.range<value>
      pdl_interp.finalize
    ^bb1:
      pdl_interp.replace %arg0 with (%1 : !pdl.value)
      %4 = ematch.dedup_region of %inlined_ops
      pdl_interp.finalize
    }
    pdl_interp.func @func_call_rewriter(%arg0: !pdl.operation, %region: !pdl_region.region, %type: !pdl.type) {
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
      pdl_interp.finalize
    }
  }