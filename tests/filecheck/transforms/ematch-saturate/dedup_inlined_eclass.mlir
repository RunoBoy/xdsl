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
      %100 = pdl_interp.apply_constraint "get_function_call"(%arg0 : !pdl.operation) : !pdl.operation -> ^bb42, ^bb1
    ^bb42:
      %110 = pdl_interp.get_attribute "sym_visibility" of %100
      %112 = pdl_interp.create_attribute "private"
      pdl_interp.are_equal %110, %112 : !pdl.attribute -> ^bb1, ^bb48
    ^bb48:
      // get region of function (function body)
      %101 = pdl_interp_region.get_region 0 of %100 : !pdl_region.region
      // check if an equivalence graph is present
      %1011 = pdl_interp_region.get_operation() called "equivalence.graph" 0 of %101
      // check if it's found
      pdl_interp.is_not_null %1011 : !pdl.operation -> ^bb421, ^bb430
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
      %1021 = pdl_interp.apply_constraint "get_arguments_of_function"(%100 : !pdl.operation) : !pdl.range<value> -> ^bb423, ^bb1
    ^bb423:
      pdl_interp.check_result_count of %arg0 is 1 -> ^bb425, ^bb1
    ^bb425:
      %1018 = pdl_interp.get_result 0 of %arg0
      %1019 = pdl_interp.get_value_type of %1018 : !pdl.type
      %1020 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%1017 : !pdl_region.region) -> (%1019 : !pdl.type)
      pdl_interp.apply_constraint "replace_func_args_with_correct_definitions"(%1020, %arg0 : !pdl.operation, !pdl.operation) -> ^bb424, ^bb1
    ^bb424:
      pdl_interp.record_match @rewriters::@func_call_rewriter(%arg0, %1020 : !pdl.operation, !pdl.operation) : benefit(1) -> ^bb1
    ^bb430:
      %102 = pdl_interp.apply_constraint "replace_return_with_yield"(%101 : !pdl_region.region) : !pdl_region.region -> ^bb43, ^bb1
    ^bb43:
      %103 = pdl_interp.apply_constraint "get_arguments_of_function"(%100 : !pdl.operation) : !pdl.range<value> -> ^bb44, ^bb1
    ^bb44:
      %104 = pdl_interp.get_result 0 of %arg0
      %105 = pdl_interp.get_value_type of %104 : !pdl.type
      %106 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%102 : !pdl_region.region) -> (%105 : !pdl.type)
      pdl_interp.apply_constraint "replace_func_args_with_correct_definitions"(%106, %100, %arg0 : !pdl.operation, !pdl.operation, !pdl.operation) -> ^bb45, ^bb1
    ^bb45:
      pdl_interp.record_match @rewriters::@func_call_rewriter(%arg0, %106 : !pdl.operation, !pdl.operation) : benefit(1) -> ^bb1
   }

  builtin.module @rewriters {
    pdl_interp.func @if_true_rewriter(%arg0 : !pdl.operation, %arg1 : !pdl.type) {
      %0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%0 : !pdl_region.region) -> (%arg1 : !pdl.type)
      %2 = pdl_interp.get_result 0 of %1
      %3 = ematch.get_class_result %2
      %4 = pdl_interp.create_range %3 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %4 : !pdl.range<value>
      pdl_interp.finalize
    }

     pdl_interp.func @if_false_rewriter(%arg0 : !pdl.operation, %arg1 : !pdl.type) {
      %0 = pdl_interp_region.get_region 1 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%0 : !pdl_region.region) -> (%arg1 : !pdl.type)
      %2 = pdl_interp.get_result 0 of %1
      %3 = ematch.get_class_result %2
      %4 = pdl_interp.create_range %3 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %4 : !pdl.range<value>
      pdl_interp.finalize
    }

    pdl_interp.func @execute_region_rewriter(%arg0: !pdl.operation) {
      %0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.inline_region %arg0 with (%0 : !pdl_region.region)
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
  }
