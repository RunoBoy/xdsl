// RUN: xdsl-opt %s -p ematch-saturate | filecheck %s

// CHECK: func.func @main(%x: i32) {
// CHECK-NEXT:   equivalence.graph : () -> () {
// CHECK-NEXT:     %a = arith.constant 2 : i32
// CHECK-NEXT:     %s = arith.constant 1 : i32
// CHECK-NEXT:     %t = equivalence.class %t_1, %y : i32
// CHECK-NEXT:     %t_1 = arith.shli %x, %s : i32
// CHECK-NEXT:     %y = arith.muli %x, %a : i32
// CHECK-NEXT:     equivalence.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   func.return
// CHECK-NEXT: }

func.func @main(%x : i32) -> () {
    equivalence.graph : () -> () {

        %u = scf.execute_region -> i32 {
            %a = arith.constant 2 : i32
            %y = arith.muli %x, %a : i32
            scf.yield %y : i32
        }

        %v = scf.execute_region -> i32 {
            %s = arith.constant 1 : i32
            %t = arith.shli %x, %s : i32
            scf.yield %t : i32
        }

        equivalence.yield
    }
    return
}


pdl_interp.func @matcher(%arg0: !pdl.operation) {
    pdl_interp.check_operation_name of %arg0 is "arith.divui" -> ^bb_div, ^bb_mul
  ^bb_fail:
    pdl_interp.finalize
  ^bb_mul:
    pdl_interp.check_operation_name of %arg0 is "arith.muli" -> ^bb_mul2, ^bb_cmp
  ^bb_mul2:
    %operand1m = pdl_interp.get_operand 1 of %arg0
    %op1m = pdl_interp.get_defining_op of %operand1m : !pdl.value
    %22 = pdl_interp.create_attribute 2 : i32
    %operand1_attrm = pdl_interp.get_attribute "value" of %op1m
    pdl_interp.are_equal %operand1_attrm, %22 : !pdl.attribute -> ^bb_mul3, ^bb_fail
  ^bb_mul3:
    pdl_interp.record_match @rewriters::@shli(%arg0 : !pdl.operation) : benefit(2), loc([]) -> ^bb_fail
  ^bb_div:
    %operand1 = pdl_interp.get_operand 1 of %arg0
    %op1 = pdl_interp.get_defining_op of %operand1 : !pdl.value
    %2 = pdl_interp.create_attribute 2 : i32
    %operand1_attr = pdl_interp.get_attribute "value" of %op1
    pdl_interp.are_equal %operand1_attr, %2 : !pdl.attribute -> ^bb_div2, ^bb_fail
  ^bb_div2:
    %operand2 = pdl_interp.get_operand 0 of %arg0
    %operand2_c = ematch.get_class_vals %operand2
    pdl_interp.foreach %op_v : !pdl.value in %operand2_c {
        %op = pdl_interp.get_defining_op of %op_v : !pdl.value
        pdl_interp.check_operation_name of %op is "arith.muli" -> ^bb1, ^bb0
      ^bb0:
        pdl_interp.continue
      ^bb1:
        %a = pdl_interp.get_operand 1 of %op
        %b = pdl_interp.get_defining_op of %a : !pdl.value
        %222 = pdl_interp.create_attribute 2 : i32
        %c = pdl_interp.get_attribute "value" of %b
        pdl_interp.are_equal %c, %222 : !pdl.attribute -> ^bb2, ^bb0
      ^bb2:
        %d = pdl_interp.get_operand 0 of %op
        %e = pdl_interp.get_defining_op of %d : !pdl.value
        pdl_interp.record_match  @rewriters::@mul_div(%arg0, %e : !pdl.operation, !pdl.operation) : benefit(2), loc([]) -> ^bb0
    } -> ^bb_fail
  ^bb_cmp:
    pdl_interp.check_operation_name of %arg0 is "scf.if" -> ^bb_cmp1, ^bb_execute_region
  ^bb_cmp1:
    %y = pdl_interp.get_operand 0 of %arg0
    %z = pdl_interp.get_defining_op of %y : !pdl.value
    %cmp_oprnd2 = pdl_interp.get_operand 1 of %z
    %cmp_opr2 = pdl_interp.get_defining_op of %cmp_oprnd2 : !pdl.value
    %2222 = pdl_interp.create_attribute 0 : i32
    %cmp_opr2_value = pdl_interp.get_attribute "value" of %cmp_opr2
    pdl_interp.are_equal %cmp_opr2_value, %2222 : !pdl.attribute -> ^bb_cmp3, ^bb_fail
  ^bb_cmp3:
    %cmp_oprnd1 = pdl_interp.get_operand 0 of %z
    %cmp_oprnd2_c = ematch.get_class_vals %cmp_oprnd1
    pdl_interp.foreach %oprnd2_v : !pdl.value in %cmp_oprnd2_c {
        %op2 = pdl_interp.get_defining_op of %oprnd2_v : !pdl.value
        pdl_interp.check_operation_name of %op2 is "arith.constant" -> ^bb1, ^bb0
      ^bb0:
        pdl_interp.continue
      ^bb1:
        %11 = pdl_interp.create_attribute 1 : i32
        %x = pdl_interp.get_attribute "value" of %op2
        pdl_interp.are_equal %x, %11 : !pdl.attribute -> ^bb2, ^bb0
      ^bb2:
        %4 = pdl_interp.get_result 0 of %op2
        %5 = pdl_interp.get_value_type of %4 : !pdl.type
        pdl_interp.record_match @rewriters::@if_true_rewriter(%arg0, %5: !pdl.operation, !pdl.type)  : benefit(2), loc([]) -> ^bb0
    } -> ^bb_fail
  ^bb_execute_region:
    pdl_interp.check_operation_name of %arg0 is "scf.execute_region" -> ^bb_ex1, ^bb_fail
  ^bb_ex1:
    pdl_interp.record_match @rewriters::@execute_region_rewriter(%arg0 : !pdl.operation) : benefit(1) -> ^bb_fail
}

module @rewriters {
    pdl_interp.func @shli(%arg0 : !pdl.operation) {
       %0 = pdl_interp.create_attribute 1 : i32
       %3 = pdl_interp.create_type i32
       %1 = pdl_interp.create_operation "arith.constant" {"value" = %0} -> (%3 : !pdl.type)

       %5 = ematch.dedup %1
       %8 = pdl_interp.get_result 0 of %5
       %15 = ematch.get_class_result %8

       %9 = pdl_interp.get_operand 0 of %arg0
       %10 = pdl_interp.create_operation "arith.shli"(%9, %15 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
       %11 = ematch.dedup %10

       %12 = pdl_interp.get_result 0 of %11
       %13 = ematch.get_class_result %12
       %14 = pdl_interp.create_range %13 : !pdl.value
       ematch.union %arg0 : !pdl.operation, %14 : !pdl.range<value>

       pdl_interp.finalize
    }

    pdl_interp.func @mul_div(%arg0 : !pdl.operation, %arg1 : !pdl.operation) {
       %12 = pdl_interp.get_result 0 of %arg1
       %13 = ematch.get_class_result %12
       %14 = pdl_interp.create_range %13 : !pdl.value
       ematch.union %arg0 : !pdl.operation, %14 : !pdl.range<value>

       pdl_interp.finalize
    }

    pdl_interp.func @if_true_rewriter(%arg0 : !pdl.operation, %arg1 : !pdl.type) {
      %0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %3 = pdl_interp_region.clone_region(%0 : !pdl_region.region)
      %7 = pdl_interp_region.region_iterator(%3 : !pdl_region.region)
      %6 = ematch.dedup_region of %7 in %arg0
      pdl_interp.is_not_null %6 : !pdl.range<operation> -> ^bb0, ^bb1
    ^bb0:
      %1 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%3 : !pdl_region.region) -> (%arg1 : !pdl.type)
      %2 = pdl_interp.get_result 0 of %1
      %4 = ematch.get_class_result %2
      %5 = pdl_interp.create_range %4 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %5 : !pdl.range<value>
      pdl_interp.finalize
    ^bb1:
      %yield_op = pdl_interp_region.get_operation() called "scf.yield" 0 of %3
      %yield_val = pdl_interp.get_operand 0 of %yield_op
      %yield_class = ematch.get_class_result %yield_val
      %yield_range = pdl_interp.create_range %yield_class : !pdl.value
      ematch.union %arg0 : !pdl.operation, %yield_range : !pdl.range<value>
      %clean = pdl_interp_region.delete_op_from_region(%yield_op : !pdl.operation) of %3
      pdl_interp.finalize
    }

    pdl_interp.func @execute_region_rewriter(%arg0: !pdl.operation) {
      %0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %1, %2 = pdl_interp_region.inline_region %arg0 with (%0 : !pdl_region.region)
      pdl_interp.replace %arg0 with (%1 : !pdl.value)
      pdl_interp.finalize
    }
}
