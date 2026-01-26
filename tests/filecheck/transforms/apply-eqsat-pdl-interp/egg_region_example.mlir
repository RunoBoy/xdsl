// RUN: xdsl-opt %s -p apply-eqsat-pdl-interp | filecheck %s

func.func @impl() -> i32 {
  // true
  %cond = arith.constant 1  : i1

  // if (true) then ... else ...
  %ifelse = scf.if %cond -> (i32) {
    %1 = arith.constant 1 : i32
    %2 = arith.constant 2 : i32
    %3 = arith.constant 3 : i32
    scf.yield %1 : i32
  } else {
    %2 = arith.constant 2 : i32
    scf.yield %2 : i32
  }
  func.return %ifelse : i32
}

pdl_interp.func @matcher(%arg0: !pdl.operation) {
  // check if operation is an if statement
  pdl_interp.check_operation_name of %arg0 is "scf.if" -> ^bb1, ^bb3
^bb1:
  // check if the conditions is a constant
  %0 = pdl_interp.get_operand 0 of %arg0
  %1 = pdl_interp.get_defining_op of %0 : !pdl.value
  pdl_interp.check_operation_name of %1 is "arith.constant" -> ^bb2, ^bb3
^bb2:
  // check if the constant is true
  %2 = pdl_interp.get_attribute "value" of %1
  %3 = pdl_interp.create_attribute 1 : i32
  pdl_interp.are_equal %2, %3 : !pdl.attribute -> ^bb3, ^bb4
^bb3:
  pdl_interp.record_match @rewriters::@pdl_generated_rewriter_1(%arg0 : !pdl.operation) : benefit(1) -> ^bb4
^bb4:
  pdl_interp.finalize
}

module @rewriters {
    pdl_interp.func @pdl_generated_rewriter_1(%arg0: !pdl.operation) {
      %0 = pdl_interp.get_region of %arg0 : !pdl.region
//      %1 = pdl_interp.get_result 0 of %arg0
//      %2 = pdl_interp.get_last_operation of %arg0
//      %3 = pdl_interp.get_operand 0 of %2
//      %4 = pdl_interp.get_operation of
//      %1 = pdl_interp.get_region_results of %0 : !pdl.range<value>
//      pdl_interp.insert_region %arg0 with (%0 : !pdl.region
//      pdl_interp.replace %arg0 with (%3: !pdl.value)
      pdl_interp.finalize
    }
}

// // if (true) then x else y -> x
// pdl.pattern : benefit(1) {
//   %x = pdl_region.region
//   %y = pdl_region.region
//   %type = pdl.type
//   %one = pdl.attribute = 1 : i32
//   %cond_true = pdl.operation "arith.constant" {"value" = %one} -> (%type : !pdl.type)
//   %true = pdl.result 0 of %cond_true
//   %original_op = pdl.region_operation "scf.if" (%true, %x, %y : !pdl.type, !pdl_region.region, !pdl_region.region) -> (%type: !pdl_region.type)
//   pdl.rewrite %original_op {
//     pdl.replace %original_op with (%x : !pdl_region.region)
//   }
