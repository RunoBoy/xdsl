func.func @compound(%arg0: f32) -> f32 {
    %0 = equivalence.graph : () -> (f32) {
      %1 = math.exp %arg0 : f32
      equivalence.yield %1 : f32
    }
    return %0 : f32
}
func.func @quant_model(%arg0: f32, %arg1: f32) -> f32 {
    %0 = equivalence.graph : () -> (f32) {
      %4 = func.call @compound(%arg0) : (f32) -> f32
      equivalence.yield %4 : f32
    }
    return %0 : f32
}


  pdl_interp.func @matcher(%arg0 : !pdl.operation) {
      pdl_interp.check_operation_name of %arg0 is "scf.execute_region" -> ^bb31, ^bb40
    ^bb1:
      pdl_interp.finalize
    ^bb31:
      pdl_interp.record_match @rewriters::@execute_region_rewriter(%arg0 : !pdl.operation) : benefit(1) -> ^bb1
    ^bb40:
      pdl_interp.check_operation_name of %arg0 is "func.call" -> ^bb41, ^bb50
    ^bb41:
      %100 = pdl_interp.apply_constraint "get_function_call"(%arg0 : !pdl.operation) : !pdl.operation -> ^bb42, ^bb1
    ^bb42:
      // get region of function (function body)
      %101 = pdl_interp_region.get_region 0 of %100 : !pdl_region.region
      // check if an equivalence graph is present
      %1011 = pdl_interp_region.get_operation() called "equivalence.graph" 0 of %101
      // check if it's found
      pdl_interp.is_not_null %1011 : !pdl.operation -> ^bb421, ^bb1
    ^bb421:
      // extract the equvialence graph body
      %1012 = pdl_interp_region.get_region 0 of %1011 : !pdl_region.region
      %1022 = pdl_interp_region.clone_region(%1012 : !pdl_region.region)
      // extract the equivalence yield
      %1013 = pdl_interp_region.get_operation() called "equivalence.yield" 0 of %1022
      pdl_interp.is_not_null %1013 : !pdl.operation -> ^bb422, ^bb1
    ^bb422:
      // create a new operation scf.yield to replace the equivalence.yield
      %1014 = pdl_interp.get_operand 0 of %1013
      %1015 = pdl_interp.create_operation "scf.yield"(%1014 : !pdl.value)
      %1016 = pdl_interp_region.insert_op_into_region(%1015 : !pdl.operation) of %1022
      %1017 = pdl_interp_region.delete_op_from_region(%1013 : !pdl.operation) of %1016
      pdl_interp.check_result_count of %arg0 is 1 -> ^bb425, ^bb1
    ^bb425:
      %1018 = pdl_interp.get_result 0 of %arg0
      %1019 = pdl_interp.get_value_type of %1018 : !pdl.type
      %1020 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%1017 : !pdl_region.region) -> (%1019 : !pdl.type)
      %1023 = ematch.dedup %1020
      %1024 = pdl_interp_region.get_region 0 of %1023 : !pdl_region.region
      %1025 = pdl_interp.get_operands of %arg0 : !pdl.range<value>
      pdl_interp.apply_constraint "replace_func_args_with_correct_definitions"(%1025, %100, %1024 : !pdl.range<value>, !pdl.operation, !pdl_region.region) -> ^bb424, ^bb1
    ^bb424:
      pdl_interp.record_match @rewriters::@func_call_rewriter(%arg0, %1023 : !pdl.operation, !pdl.operation) : benefit(1) -> ^bb1
    ^bb50:
      pdl_interp.check_operation_name of %arg0 is "math.exp" -> ^bb51, ^bb1
    ^bb51:
      pdl_interp.check_operand_count of %arg0 is 1 -> ^bb52, ^bb1
    ^bb52:
      %200 = pdl_interp.get_operand 0 of %arg0
      pdl_interp.is_not_null %200 : !pdl.value -> ^bb53, ^bb1
    ^bb53:
      %201 = pdl_interp.get_defining_op of %200 : !pdl.value
      pdl_interp.is_not_null %201 : !pdl.operation -> ^bb54, ^bb1
    ^bb54:
      pdl_interp.check_operation_name of %201 is "arith.addf" -> ^bb55, ^bb1
    ^bb55:
      %202 = pdl_interp.get_operand 0 of %201
      %204 = ematch.get_class_vals %202
      pdl_interp.foreach %205 : !pdl.value in %204 {
        %206 = pdl_interp.get_defining_op of %205 : !pdl.value
        pdl_interp.is_not_null %206 : !pdl.operation -> ^bb57, ^bb56
        ^bb56:
          pdl_interp.continue
        ^bb57:
          pdl_interp.check_operation_name of %206 is "math.log" -> ^bb58, ^bb56
        ^bb58:
          %207 = pdl_interp.get_operand 1 of %201
          %208 = ematch.get_class_vals %207
          pdl_interp.foreach %209 : !pdl.value in %208 {
            %210 = pdl_interp.get_defining_op of %209 : !pdl.value
            pdl_interp.is_not_null %210 : !pdl.operation -> ^bb60, ^bb59
            ^bb59:
              pdl_interp.continue
            ^bb60:
              pdl_interp.check_operation_name of %210 is "math.log" -> ^bb61, ^bb59
            ^bb61:
              %211 = pdl_interp.get_result 0 of %arg0
              %212 = ematch.get_class_result %211
              %213 = pdl_interp.get_value_type of %212 : !pdl.type
              pdl_interp.record_match @rewriters::@sum_of_logs(%arg0, %206, %210,  %213 : !pdl.operation, !pdl.operation, !pdl.operation, !pdl.type) : benefit(1) -> ^bb59

          } -> ^bb56
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
      ematch.dedup_region %inlined_ops
      pdl_interp.replace %arg0 with (%1 : !pdl.value)
      pdl_interp.finalize
    }

    pdl_interp.func @func_call_rewriter(%arg0 : !pdl.operation, %arg1 : !pdl.operation) {
      %0 = pdl_interp.get_result 0 of %arg1
      %1 = ematch.get_class_result %0
      %2 = pdl_interp.create_range %1 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %2 : !pdl.range<value>
      pdl_interp.finalize
    }

    pdl_interp.func @sum_of_logs(%arg0 : !pdl.operation, %arg1 : !pdl.operation, %arg2 : !pdl.operation, %arg3 : !pdl.type) {
      %0 = pdl_interp.get_result 0 of %arg0

      %x = pdl_interp.get_operand 0 of %arg1
      %y = pdl_interp.get_operand 0 of %arg2

      %1 = pdl_interp.create_operation "arith.mulf"(%x, %y : !pdl.value, !pdl.value) -> (%arg3 : !pdl.type)
      %11 = ematch.dedup %1
      %2 = pdl_interp.get_result 0 of %11
      %3 = ematch.get_class_result %2

      %4 = pdl_interp.create_range %3 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %4 : !pdl.range<value>

      pdl_interp.finalize
    }
  }
