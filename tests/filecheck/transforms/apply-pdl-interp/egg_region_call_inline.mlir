// RUN: xdsl-opt %s -p apply-eqsat-pdl-interp | filecheck %s

func.func @compute_value(%cond: i1) -> i32 {
%ifelse = scf.if %cond -> (i32) {
  %1 = arith.constant 1 : i32
  scf.yield %1 : i32
} else {
  %2 = arith.constant 2 : i32
  scf.yield %2 : i32
}
func.return %ifelse : i32
}

func.func @impl() -> i32 {
%cond = arith.constant true
%test = func.call @compute_value(%cond) : (i1) -> i32
func.return %test : i32
}

pdl_interp.func @matcher(%arg0: !pdl.operation) {
  // check if operation is an if statement
  pdl_interp.check_operation_name of %arg0 is "func.call" -> ^bb1, ^bb2
^bb1:
  pdl_interp.debug_print "Matched func.call operation"
  %0 = pdl_interp.get_function_of_call %arg0
  pdl_interp.record_match @rewriters::@pdl_generated_rewriter_1(%arg0, %0 : !pdl.operation, !pdl.operation) : benefit(1) -> ^bb2
^bb2:
  pdl_interp.finalize
}

module @rewriters {
    pdl_interp.func @pdl_generated_rewriter_1(%arg0: !pdl.operation) {
      pdl_interp.finalize
    }
}
